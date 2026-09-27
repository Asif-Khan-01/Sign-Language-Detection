import os, json, time, collections
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils  as mp_drawing
from mediapipe.tasks.python.vision import drawing_styles as mp_draw_styles
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

model       = load_model('models/best_lstm_model.keras')
meta        = json.load(open('models/model_metadata.json'))
cfg         = json.load(open('processed_data/sequence_config.json'))
CLASS_NAMES = meta['class_names']
SEQ_LEN     = cfg['SEQUENCE_LENGTH']
FEAT_SIZE   = cfg['FEATURE_SIZE']
THRESHOLD   = meta.get('confidence_threshold', 0.7)
HAND_CONN   = HandLandmarksConnections.HAND_CONNECTIONS

def make_landmarker():
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opts)

def get_landmarks(frame, landmarker, ts_ms):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
    lm, found, ann = np.zeros(FEAT_SIZE, np.float32), False, frame.copy()
    if result.hand_landmarks:
        found = True
        for hi, hand in enumerate(result.hand_landmarks[:2]):
            mp_drawing.draw_landmarks(ann, hand, HAND_CONN,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())
            off = hi * 63
            for li, l in enumerate(hand):
                lm[off + li*3], lm[off + li*3+1], lm[off + li*3+2] = l.x, l.y, l.z
    return lm, found, ann

def run_detector(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'Cannot open camera {camera_id}')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buffer    = collections.deque(maxlen=SEQ_LEN)
    prob_hist = collections.deque(maxlen=10)
    sentence  = []
    fps_q     = collections.deque(maxlen=30)
    t0        = time.time()
    t0_ms     = int(t0 * 1000)
    prev_t    = t0
    last_hand = False
    pred_word, pred_conf = None, 0.0

    with make_landmarker() as lm_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame  = cv2.flip(frame, 1)
            ts_ms  = int(time.time() * 1000) - t0_ms
            lm_vec, hand_found, ann = get_landmarks(frame, lm_detector, ts_ms)

            if last_hand and not hand_found:
                buffer.clear()
                prob_hist.clear()
            last_hand = hand_found
            buffer.append(lm_vec)

            if len(buffer) == SEQ_LEN:
                seq   = np.expand_dims(np.array(buffer, np.float32), 0)
                probs = model.predict(seq, verbose=0)[0]
                prob_hist.append(probs)
                avg   = np.mean(list(prob_hist), axis=0)
                idx   = int(np.argmax(avg))
                pred_word, pred_conf = CLASS_NAMES[idx], float(avg[idx])

            now = time.time()
            fps_q.append(1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            fps    = np.mean(fps_q)

            h, w = ann.shape[:2]
            cv2.rectangle(ann, (0, 0), (w, 60), (0, 0, 0), -1)
            color = (0, 255, 0) if hand_found else (0, 0, 255)
            cv2.putText(ann, 'HAND ON' if hand_found else 'HAND OFF',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(ann, f'FPS: {fps:.1f}',
                        (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if pred_word and pred_conf >= THRESHOLD:
                cv2.putText(ann, f'{pred_word}  {pred_conf*100:.0f}%',
                            (10, h - 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 100), 3)
            if sentence:
                cv2.putText(ann, 'Sentence: ' + ' '.join(sentence[-6:]),
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(ann, 'A:Add  C:Clear  Q:Quit',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('ASL Real-Time Detector', ann)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('a') and pred_word and pred_conf >= THRESHOLD:
                if not sentence or sentence[-1] != pred_word:
                    sentence.append(pred_word)
                    print(f'Added: {pred_word}  |  Sentence: {sentence}')
            elif key == ord('c'):
                sentence.clear()

    cap.release()
    cv2.destroyAllWindows()
    print('Final sentence:', ' '.join(sentence) if sentence else '(empty)')

if __name__ == '__main__':
    run_detector()
