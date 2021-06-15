import os
import pybullet_data
from environments.locomotion.scene_abstract import Scene
import pybullet as p


class StadiumScene(Scene):
    zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    stadiumLoaded = 0

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
        if (self.stadiumLoaded == 0):
            self.stadiumLoaded = 1

            # stadium_pose = cpp_household.Pose()
            # if self.zero_at_running_strip_start_line:
            #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
            if self.enable_grid:
                filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            else:
                filename = os.path.join("environments/locomotion/assets", "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            # filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
            # self.ground_plane_mjcf = self._p.loadSDF(filename)
            #
            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
                self._p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, i)

            #	for j in range(p.getNumJoints(i)):
            #		self._p.changeDynamics(i,j,lateralFriction=0)
            # despite the name (stadium_no_collision), it DID have collision, so don't add duplicate ground


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class MultiplayerStadiumScene(StadiumScene):
    multiplayer = True
    players_count = 3

    def actor_introduce(self, robot):
        StadiumScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)
