"""
Main Entrypoint: Real-Time Physiological Liveness Verification System.
Includes:
- Dynamic FPS calibration from timestamps
- Multi-ROI convex hull tracking
- Adaptive sensitivity settings ('relaxed', 'normal', 'strict')
- Interactive key commands
"""

import argparse
import time
import os
import sys
import cv2
import numpy as np
from collections import deque

from src.core.face_tracker import FaceTracker
from src.core.rppg_extractor import RPPGExtractor
from src.core.liveness_verifier import LivenessVerifier, LivenessVerdict
from src.ui.dashboard import Dashboard

def run_pipeline(
    source=0,
    algo="pos",
    buffer_size=150,
    sensitivity="normal",
    synthetic_demo=False
):
    print("==================================================================")
    print("      DEEPFAKE-PROOF LIVE IDENTITY VERIFICATION (rPPG)            ")
    print("==================================================================")
    print(f"[*] Initializing Face Tracker (MediaPipe FaceLandmarker)...")
    tracker = FaceTracker()
    
    print(f"[*] rPPG Extraction Algorithm: {algo.upper()}")
    extractor = RPPGExtractor(method=algo, buffer_size=buffer_size)

    print(f"[*] Liveness Verifier (Buffer: {buffer_size} frames, Sensitivity: {sensitivity.upper()})...")
    verifier = LivenessVerifier(fs=30.0, min_frames=75, buffer_size=buffer_size, sensitivity=sensitivity)

    dashboard = Dashboard(width=1000, height=700)

    # Buffers to store RGB time-series and timestamps
    buf_comb = deque(maxlen=buffer_size)
    buf_fh = deque(maxlen=buffer_size)
    buf_lc = deque(maxlen=buffer_size)
    buf_rc = deque(maxlen=buffer_size)
    time_buf = deque(maxlen=buffer_size)

    # Handle synthetic demo or video source
    if synthetic_demo:
        print("[*] Running in SYNTHETIC SIMULATOR mode...")
        cap = None
    else:
        try:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
        except ValueError:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[!] ERROR: Could not open video source: {source}")
            print("    TIP: Run with '--synthetic-demo' to test without a physical camera!")
            return

    fps_tracker = deque(maxlen=30)
    frame_idx = 0
    current_sensitivity = sensitivity
    sensitivities = ["normal", "relaxed", "strict"]

    print("\n[Interactive Controls]:")
    print("  'q' or ESC : Quit")
    print("  'c'        : Cycle sensitivity (NORMAL -> RELAXED -> STRICT)")
    print("  's'        : Save snapshot report")
    print("  'r'        : Reset buffer to recalibrate\n")

    while True:
        t_start = time.time()
        frame_idx += 1

        if synthetic_demo:
            frame = np.zeros((700, 1000, 3), dtype=np.uint8)
            frame[:] = (35, 30, 25)
            cv2.ellipse(frame, (500, 360), (140, 190), 0, 0, 360, (180, 200, 220), -1)

            # Switch between real human (72 BPM) and deepfake every 200 frames
            if (frame_idx // 200) % 2 == 0:
                demo_type = "REAL HUMAN (Simulated 72 BPM)"
                pulse_val = 2.5 * np.sin(2 * np.pi * 1.20 * (frame_idx / 30.0))
                fh_rgb = np.array([120.0, 130.0 + pulse_val, 110.0])
                lc_rgb = np.array([122.0, 131.0 + pulse_val * 0.95, 112.0])
                rc_rgb = np.array([119.0, 129.0 + pulse_val * 0.98, 109.0])
            else:
                demo_type = "DEEPFAKE / SPOOF (Asynchronous pixel noise)"
                fh_rgb = np.array([120.0, 130.0 + 3.0 * np.sin(2 * np.pi * 0.4 * frame_idx / 30.0), 110.0])
                lc_rgb = np.array([122.0, 131.0 + np.random.normal(0, 2.0), 112.0])
                rc_rgb = np.array([119.0, 129.0 + 3.0 * np.sin(2 * np.pi * 2.3 * frame_idx / 30.0), 109.0])

            comb_rgb = (fh_rgb + lc_rgb + rc_rgb) / 3.0
            face_result = None
            
            buf_comb.append(comb_rgb)
            buf_fh.append(fh_rgb)
            buf_lc.append(lc_rgb)
            buf_rc.append(rc_rgb)
            time_buf.append(time.time())

            cv2.putText(frame, f"[DEMO SIMULATOR]: {demo_type}", (20, 135),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 200, 80), 1, cv2.LINE_AA)
            time.sleep(0.02)
        else:
            ret, frame = cap.read()
            if not ret:
                print("[*] End of video stream.")
                break

            # Detect face & extract ROIs
            face_result = tracker.process_frame(frame)

            if face_result is not None:
                buf_comb.append(face_result.combined_rgb)
                buf_fh.append(face_result.forehead_rgb)
                buf_lc.append(face_result.left_cheek_rgb)
                buf_rc.append(face_result.right_cheek_rgb)
                time_buf.append(time.time())

        # Compute dynamic actual FPS from timestamp history
        if len(time_buf) > 10:
            duration = time_buf[-1] - time_buf[0]
            actual_fs = (len(time_buf) - 1) / duration if duration > 0.05 else 30.0
        else:
            actual_fs = 30.0

        # Run extraction & verification
        if len(buf_comb) > 10:
            p_comb = extractor.extract(np.array(buf_comb))
            p_fh = extractor.extract(np.array(buf_fh))
            p_lc = extractor.extract(np.array(buf_lc))
            p_rc = extractor.extract(np.array(buf_rc))

            verdict = verifier.verify(p_comb, p_fh, p_lc, p_rc, fs=actual_fs)
        else:
            verdict = LivenessVerdict(
                is_live=False,
                liveness_score=0.0,
                bpm=0.0,
                snr_db=-10.0,
                spatial_correlation=0.0,
                regional_correlations={"FL": 0.0, "FR": 0.0, "LR": 0.0},
                status_label="NO_FACE" if not synthetic_demo and face_result is None else "ANALYZING",
                reason="Position face in camera view and remain still...",
                filtered_pulse=np.zeros(1),
                buffer_progress=0.0
            )

        # FPS calculation
        t_cost = max(1e-4, time.time() - t_start)
        fps_tracker.append(1.0 / t_cost)
        avg_fps = sum(fps_tracker) / len(fps_tracker)

        # Compose HUD Canvas
        canvas = dashboard.render(frame, face_result, verdict, fps=avg_fps, sensitivity=current_sensitivity)

        cv2.imshow("Bio-Verify // Deepfake-Proof rPPG Liveness", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            curr_idx = sensitivities.index(current_sensitivity)
            current_sensitivity = sensitivities[(curr_idx + 1) % len(sensitivities)]
            verifier.set_sensitivity(current_sensitivity)
            print(f"[*] Sensitivity mode changed to: {current_sensitivity.upper()}")
        elif key == ord('r'):
            buf_comb.clear()
            buf_fh.clear()
            buf_lc.clear()
            buf_rc.clear()
            time_buf.clear()
            print("[*] Buffer reset. Recalibrating...")
        elif key == ord('s'):
            filename = f"liveness_report_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"[+] Saved verification report snapshot: {filename}")

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("[*] Pipeline shutdown complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake-Proof Live Identity Verification via rPPG")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0) or path to video file")
    parser.add_argument("--algo", type=str, default="pos", choices=["pos", "chrom", "green"], help="rPPG extraction algorithm")
    parser.add_argument("--buffer", type=int, default=150, help="Buffer size in frames (default 150 ~ 5s)")
    parser.add_argument("--sensitivity", type=str, default="normal", choices=["relaxed", "normal", "strict"], help="Detection sensitivity threshold")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run interactive simulator mode")
    
    args = parser.parse_args()
    run_pipeline(
        source=args.source,
        algo=args.algo,
        buffer_size=args.buffer,
        sensitivity=args.sensitivity,
        synthetic_demo=args.synthetic_demo
    )
