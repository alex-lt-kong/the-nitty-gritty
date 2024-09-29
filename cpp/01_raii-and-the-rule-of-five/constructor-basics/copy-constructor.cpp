
#include <iostream>
#include <utility>
using namespace std;

class Point {
private:
  int x, y;
  int *ptr;

public:
  Point(int x1, int y1) {
    cout << "normal constructor is called (" << x1 << ", " << y1 << ")\n";
    x = x1;
    y = y1;
  }
  Point(pair<int, int> xy) {
    cout << "normal constructor is called (" << xy.first << ", " << xy.second
         << ")\n";
    x = xy.first;
    y = xy.second;
  }

  // Copy constructor
  Point(const Point &p1) {
    x = p1.x;
    y = p1.y;
    cout << "copy constructor is called (" << p1.x << ", " << p1.y << ")\n";
  }

  int getX() { return x; }
  int getY() { return y; }
};

int main() {
  Point p1(10, 15);                    // Normal constructor is called here
  auto p2 = Point(3, 14);              // Normal constructor is called here
  Point p3 = pair<int, int>(123, 456); // Normal constructor is called here
  Point p4 = p1;                       // Copy constructor is called here
  p3 = p2; // assignment operator is called here (beyond the scope of this
           // sample)

  // Let us access values assigned by constructors
  cout << "p1.x = " << p1.getX() << ", p1.y = " << p1.getY() << "\n";
  cout << "p2.x = " << p3.getX() << ", p2.y = " << p3.getY() << endl;
  return 0;
}