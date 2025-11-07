# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import Eelume.PyPost as pp

class SunstoneKfAidingCreator(object):
    def __init__(self,digest_filehandler : pp.SunstoneFileHandler,cp_file_handler : pp.SunstoneFileHandler):
        self.main_handler = digest_filehandler
        self.kf_handler = cp_file_handler
        
    def __create_aiding(self, nav_signal, time_of_last_input, time_of_last_aiding, sensor_signal, names, include_last_input=True):
        assert type(nav_signal) is tuple
        assert type(sensor_signal) is tuple
        assert(len(nav_signal) == len(sensor_signal) == len(names))

        N = len(nav_signal)
        output = ()

        for i in range(N):
            output += (nav_signal[i],)
            if include_last_input:
                output += (pp.find_values_based_on_kf_status_timestamps(time_of_last_input,sensor_signal[i],'KfStatus-input/' + names[i]),)
            output += (pp.find_values_based_on_kf_status_timestamps(time_of_last_aiding, sensor_signal[i], 'KfStatus-aiding/' + names[i]),)

        return output

    def create_horisontal_position_aiding(self, nav_pos, sensor_pos):       
        return self.__create_aiding(nav_pos, 
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastPosInput'),
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastPosAiding'),
                    sensor_pos,
                    ('north','east')
        )

    def create_depth_aiding(self, nav_depth, sensor_depth):       
        return self.__create_aiding(nav_depth, 
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastDepthInput'),
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastDepthAiding'),
                    sensor_depth,
                    ('depth',)
        )

    def create_attitude_aiding(self,nav_attitude):        
        return self.__create_aiding(nav_attitude,
                self.kf_handler.get_signal('KFStatus', 'timeOfLastOrientationInput'),
                self.kf_handler.get_signal('KFStatus', 'timeOfLastOrientationAiding'),
                nav_attitude,
                ('roll','pitch','yaw'))

    def create_rel_vel_aiding(self, nav_vel, sensor_rel_vel):
        return self.__create_aiding(nav_vel,
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastRelVelInput'),
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastRelVelAiding'),
                    sensor_rel_vel,
                    ('rel_vel/surge','rel_vel/sway','rel_vel/heave'))
                    
    def create_abs_vel_aiding(self, nav_vel, sensor_abs_vel):
        return self.__create_aiding(nav_vel,
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastAbsVelInput'),
                    self.kf_handler.get_signal('KFStatus', 'timeOfLastAbsVelAiding'),
                    sensor_abs_vel,
                    ('abs_vel/surge','abs_vel/sway','abs_vel/heave'))                    