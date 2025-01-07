# -*- coding: utf-8 -*-
# @Auther   : Zheng SUN (ZainZh)
# @Time     : 2025/1/7
# @Address  : clover Lab @ CUHK
# @FileName : curi_sim.py

# @Description : TODO

from base_sim import RobotSim


class CURISim(RobotSim):
    def __init__(self, config="curi"):
        super().__init__(config)

    def interact_with_curi(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)


if __name__ == "__main__":
    curi_sim = CURISim()
    curi_sim.interact_with_curi()
