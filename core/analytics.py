from shapely.geometry import LineString, Point

class SpatialAnalyticsEngine:
    def __init__(self, line_coords: list = None):
        self.line = LineString(line_coords) if line_coords else LineString([(0, 240), (640, 240)])
        self.crossed_ids = set()
        self.cross_count = 0

    def check_line_crossing(self, track_id: int, bbox: list):
        x1, y1, x2, y2 = bbox[:4]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        if track_id not in self.crossed_ids:
            if abs(cy - self.line.coords[0][1]) < 10:
                self.crossed_ids.add(track_id)
                self.cross_count += 1
        return self.cross_count