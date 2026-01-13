#!/usr/bin/env python3
import time
import cv2
from car import Car
from mapping import render_occupancy_map

def main():
    car = Car()
    car.set_motors(0,0,0,0)

    # freeze pose at origin
    car.pose.x_cm = 0.0
    car.pose.y_cm = 0.0
    car.pose.th_deg = 0.0

    cv2.namedWindow("Map", cv2.WINDOW_NORMAL)

    try:
        for _ in range(40):
            angs, dists = car.scan_distances(angles=(30,60,90,120,150), settle=0.06, samples=7)
            car.mem.update_from_scan(car.pose, angs, dists, max_range_cm=120.0)
            img = render_occupancy_map(car.mem, car.pose, scale=8, pad=10, draw_grid=True)
            cv2.imshow("Map", img)
            cv2.waitKey(1)
            time.sleep(0.05)
    finally:
        car.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
