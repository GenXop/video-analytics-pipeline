import numpy as np

class IdentityTracker:
    def __init__(self):
        self.active_tracks = {}
        self.next_id = 1

    def update(self, detections: np.ndarray):
        tracked_objects = []
        updated_tracks = {}

        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, cls_id = det[:6]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            matched_id = None
            min_dist = float('inf')
            
            for t_id, pos in self.active_tracks.items():
                dist = np.hypot(cx - pos[0], cy - pos[1])
                if dist < 50 and dist < min_dist:
                    matched_id = t_id
                    min_dist = dist

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            updated_tracks[matched_id] = (cx, cy)
            tracked_objects.append([int(x1), int(y1), int(x2), int(y2), matched_id, int(cls_id)])

        self.active_tracks = updated_tracks
        return tracked_objects