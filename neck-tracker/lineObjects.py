import numpy as np

class Vec2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line:
    def __init__(self,m,c):
        self.m = m 
        self.c = c

        self.angle = np.arctan(m)
        self.is_vertical = self.m == float('inf')
    
    @classmethod
    def from_points(cls,x1,y1,x2,y2):
        if x2-x1 == 0:
            return cls(float('inf'),x1)

        m = (y2-y1)/(x2-x1)
        c = y1-m*x1 
        return cls(m,c)

    def does_intersect(self,other):
        if type(other) == Vec2d:
            return self.m * other.x + self.c == other.y

        if type(other) == Line:
            return self.m != other.m 

        raise Exception(f"does_intersect not implemented for type {other.__name__}")

    def get_intersection(self, other):
        if type(other) == Vec2d:
            return other

        if type(other) == Line:
            if not self.does_intersect(other):
                return None
            
            x = (self.c - other.c)/(other.m - self.m)
            y = self.m*x + self.c 

            return Vec2d(x,y)

        if type(other) == LineSegment:
            potentialIntersection = self.get_intersection(other.line)
            if potentialIntersection == None:
                return None

            in_bound = (potentialIntersection.x >= other.x1 and potentialIntersection.x <= other.x2)

            if in_bound:
                return potentialIntersection
            
            return None


        raise Exception(f"get_intersection not implemented for type {other.__name__}")
    
    def point_distance(self, pos):
        if self.is_vertical:
            return abs(pos.x - self.c)
        return abs(self.m * pos.x - pos.y + self.c) / np.sqrt(self.m**2 + 1)

class LineSegment:
    def __init__(self,x1,y1,x2,y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

        self.line = Line.from_points(x1,y1,x2,y2)

    def does_intersect(self,other):
        if type(other) == Vec2d:
            if self.x1 >= other.x and self.x2 <= other.x:
                return self.line.does_intersect(other)

            return False

        if type(other) == Line:
            return other.does_intersect(self) 

        raise Exception(f"does_intersect not implemented for type {other.__name__}")

    def get_intersection(self, other):
        if type(other) == Vec2d:
            if self.does_intersect(other):
                return other

            return None

        if type(other) == Line:
            return other.get_intersection(self)

        if type(other) == LineSegment:
            potentialIntersection = other.line.get_intersection(self)
            if potentialIntersection == None:
                return None

            in_bound = (potentialIntersection.x >= other.x1 and potentialIntersection.x <= other.x2)

            if in_bound:
                return potentialIntersection
            
            return None


        raise Exception(f"get_intersection not implemented for type {other.__name__}")

    def closest_point_on_segment(self, pos):
        """Find closest point on segment to given point"""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length_sq = dx**2 + dy**2
        
        if length_sq == 0:
            return Vec2d(self.x1, self.y1)
        
        t = max(0, min(1, ((pos.x - self.x1) * dx + (pos.y - self.y1) * dy) / length_sq))
        return Vec2d(self.x1 + t * dx, self.y1 + t * dy)
    
    def point_on_segment(self, x, y):
        """Check if point (x,y) lies within the segment bounds"""
        min_x, max_x = min(self.x1, self.x2), max(self.x1, self.x2)
        min_y, max_y = min(self.y1, self.y2), max(self.y1, self.y2)
        return min_x <= x <= max_x and min_y <= y <= max_y
    

def point_to_point_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def min_distance(l1, l2):
    """Minimum distance between two line segments"""
    # Check if segments intersect
    intersection = l1.get_intersection(l2)
    if intersection and l1.point_on_segment(intersection.x, intersection.y) and l2.point_on_segment(intersection.x, intersection.y):
        return 0.0
    
    # Check distances from each endpoint to the other segment
    p1 = Vec2d(l1.x1, l1.y1)
    p2 = Vec2d(l1.x2, l1.y2)
    p3 = Vec2d(l2.x1, l2.y1)
    p4 = Vec2d(l2.x2, l2.y2)
    
    distances = [
        point_to_point_dist(p1, l2.closest_point_on_segment(p1)),
        point_to_point_dist(p2, l2.closest_point_on_segment(p2)),
        point_to_point_dist(p3, l1.closest_point_on_segment(p3)),
        point_to_point_dist(p4, l1.closest_point_on_segment(p4)),
    ]
    
    return min(distances)
