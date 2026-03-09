import cv2
from cam import get_head_counts

while True:

    z1, z2, z3, z4 = get_head_counts()

    print("Zone1:",z1,"Zone2:",z2,"Zone3:",z3,"Zone4:",z4)

    # important to avoid window freeze
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()