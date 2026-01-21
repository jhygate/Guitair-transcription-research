#include <iostream>
#include <cmath>
#include <utility>

class Vec2d{
public:
  double x,y;
  Vec2d(double x_val, double y_val): x(x_val), y(y_val) {}
};

class LineSegment {
public:
  double m,c,x1,x2;
  double y1,y2;

  LineSegment(double m_val, double c_val, double x1_val, double x2_val): m(m_val), c(c_val), x1(x1_val), x2(x2_val) {
    y1 = m*x1 + c;
    y2 = m*x2 +c;
  }

  Vec2d intersection(LineSegment l){
    double x = (l.c - c)/(m-l.m);
    double y = ((c*l.m)-(m*l.c))/(l.m - m);
    return Vec2d(x,y);
  }

  bool intersects(LineSegment l){
    Vec2d res = intersection(l);
    if (res.x >= x1 and res.x <= x2){
      return true;
    }
    return false;
  }

  double pointDistance(Vec2d pos){
    return (m*pos.x - pos.y + c)/(sqrt(m*m + 1));
  }

};

double minDist(LineSegment l1,LineSegment l2){
  if (l1.intersects(l2)){
    return 0.0;
  }
  
  Vec2d l1p1 = Vec2d(l1.x1, l1.y1);
  Vec2d l1p2 = Vec2d(l1.x2, l1.y2);

  Vec2d l2p1 = Vec2d(l2.x1, l2.y1);
  Vec2d l2p2 = Vec2d(l2.x2, l2.y2);

  double l1p1tol2 = l2.pointDistance(l1p1);
  double l1p2tol2 = l2.pointDistance(l1p2);

  double l2p1tol1 = l1.pointDistance(l2p1);
  double l2p2tol1 = l1.pointDistance(l2p2);

  return std::min({abs(l1p1tol2), abs(l1p2tol2), abs(l2p1tol1), abs(l2p2tol1)});
}

int main(){
  LineSegment line1(2.0,0.0,0.0,1.0);
  LineSegment line2(10.0,0.0,0.0,1.0);
  
  Vec2d l1l2intersection = line1.intersection(line2);
  std::cout << "Line with slope" << minDist(line1,line2) << std::endl;
  return 0;
}
