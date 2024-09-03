import numpy as np
from typing import Tuple, List, Optional

class Point:
    def __init__(self, x: float, y: float, w: float = 1.0):
        """
        Initialize a point in homogeneous coordinates.
        If `w` is 1, the point is in Cartesian (physical) coordinates.
        """
        self.x = x
        self.y = y
        self.w = w

    @classmethod
    def from_cartesian(cls, x: float, y: float):
        """
        Create a point from Cartesian coordinates.
        """
        return cls(x, y, 1.0)

    def to_cartesian(self) -> Tuple[float, float]:
        """
        Convert the point to Cartesian coordinates.
        Returns a tuple (x, y).
        """
        if self.w != 0:
            return (self.x / self.w, self.y / self.w)
        else:
            raise ValueError("Cannot convert point at infinity to Cartesian coordinates.")

    def to_homogeneous(self) -> Tuple[float, float, float]:
        """
        Return the homogeneous coordinates as a tuple (x, y, w).
        """
        return (self.x, self.y, self.w)

    def cross_product(self, other: 'Point') -> 'Point':
        """
        Compute the cross product of this point with another point.
        Returns the result as a new Point.
        """
        x = self.y * other.w - self.w * other.y
        y = self.w * other.x - self.x * other.w
        w = self.x * other.y - self.y * other.x
        return Point(x, y, w)

    def is_collinear(self, p1: 'Point', p2: 'Point') -> bool:
        """
        Check if three points (self, p1, p2) are collinear.
        """
        return np.linalg.det(np.array([
            [self.x, self.y, self.w],
            [p1.x, p1.y, p1.w],
            [p2.x, p2.y, p2.w]
        ])) == 0

    def distance_to(self, other: 'Point') -> float:
        """
        Calculate the Euclidean distance between two points in Cartesian space.
        """
        if self.w != 0 and other.w != 0:
            x1, y1 = self.to_cartesian()
            x2, y2 = other.to_cartesian()
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        else:
            raise ValueError("Distance cannot be calculated for points at infinity.")

    def to_line(self, other: 'Point') -> 'Line':
        """
        Create a line passing through this point and another point.
        """
        return Line.from_points(self, other)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, w={self.w})"


class Line:
    def __init__(self, a: float, b: float, c: float):
        """
        Initialize a line in homogeneous coordinates.
        Line is defined by ax + by + c = 0.
        """
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def from_points(cls, p1: Point, p2: Point):
        """
        Create a line from two points in homogeneous coordinates.
        """
        line = p1.cross_product(p2)
        return cls(line.x, line.y, line.w)

    def intersection_with(self, other: 'Line') -> Optional[Point]:
        """
        Find the intersection point of this line with another line.
        Returns the intersection point in homogeneous coordinates.
        """
        intersection = Point(self.a, self.b, self.c).cross_product(Point(other.a, other.b, other.c))
        if intersection.w != 0:  # Intersection is not at infinity
            return intersection
        else:
            return None  # Lines are parallel or coincident

    def is_point_on_line(self, point: Point) -> bool:
        """
        Check if a given point lies on the line.
        """
        return np.isclose(self.a * point.x + self.b * point.y + self.c * point.w, 0)

    def __repr__(self):
        return f"Line(a={self.a}, b={self.b}, c={self.c})"


# Example usage:

print(Point(2,3).to_homogeneous().to_line(Point(2,1).to_homogeneous()))

# Creating points in Cartesian space
p1 = Point.from_cartesian(1, 2)
p2 = Point.from_cartesian(4, 5)
p3 = Point.from_cartesian(7, 8)

# Convert points to homogeneous coordinates
p1_homogeneous = p1.to_homogeneous()
print(f"p1 in homogeneous coordinates: {p1_homogeneous}")

# Create a line from two points
line = p1.to_line(p2)
print(f"Line created from p1 and p2: {line}")

# Check if p3 is on the line
is_on_line = line.is_point_on_line(p3)
print(f"Is p3 on the line? {is_on_line}")

# Find the intersection of two lines
p4 = Point.from_cartesian(2, 3)
p5 = Point.from_cartesian(5, 7)
line2 = p4.to_line(p5)
intersection = line.intersection_with(line2)
if intersection:
    print(f"Intersection point of the two lines: {intersection.to_cartesian()}")
else:
    print("The lines do not intersect (they are parallel or coincident).")
