"""Stand or lie the Go2, following visual_nav's own safety sequence.

Order matters and is not mine: the D1 is latched while the robot is still PRONE and
before sport mode is selected, because latch_arm only ever holds joints where they
already are, and standing is what puts the arm's moment onto the hind legs.
"""
import sys
sys.path.insert(0, "/home/unitree/robotics-connect/unitree/go2/visual_nav")
sys.path.insert(0, "/home/unitree/robotics-connect")
from safety import ArmStowMonitor, latch_arm, lie_down, stand_up
from unitree.go2.locomotion.go2_locomotion import Go2Locomotion

want = sys.argv[1]
loco = Go2Locomotion(iface="eth0")
loco.connect()          # this is what initialises DDS, not the constructor
arm = ArmStowMonitor(); arm.start()
latch = latch_arm(arm, iface="eth0")
print("latch:", latch)
if not latch.held:
    raise SystemExit("REFUSING: the D1 latch did not take — hand-pose it flat and retry")
if want == "stand":
    stand_up(loco); print("STANDING")
else:
    lie_down(loco); print("PRONE")
