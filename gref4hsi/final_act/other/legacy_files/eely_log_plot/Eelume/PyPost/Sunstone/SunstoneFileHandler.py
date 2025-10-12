# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

import Eelume.PyPost as pp
import h5py
import numpy as np
import pandas as pd
import os

def create_time_stamps(timeraw, timeunit = 'us'):
    """
    Converts Sunstone "raw time" to datetime
    """
    
    datetimes = pd.to_datetime(timeraw, unit = timeunit)
    time = np.empty(len(datetimes),dtype=np.double)
    for i in range(time.size): # Should be optimized and get rid of python for loop
        time[i] = datetimes[i].timestamp()
    return time

class SunstoneFileHandler(object):
    """ 
    Class that handles reading of data from a Sunstone log files.
    """
    
    class HdfGroupNamer(object):
        def __init__(self, path : str, time_field_name : str, pretty_name=None, time_unit = 'us'):
            """
            Data structure that holds information about a specific h5 path to load from file.

            Parameters
            ----------
            path : str
                The h5 path, e.g. DatagramBase/CalculatedData/NavigationSolutionData/slotID0/
            time_field_name : str
                Field that holds timestamps
            pretty_name : str, The default is None.
                A prettyname used for presentation, typically in plots. If None is name set to pathname 
            time_unit : str. The default is 'us'.

            Returns
            -------
            None.

            """
            self.path = path
            self.time_field_name = time_field_name  
            self.name = pretty_name if pretty_name is not None else path
            self.time_unit = time_unit
            
    class GroupData(object):
        def __init__(self,time,data,group_info):
            self.time = time
            self.data = data
            self.group_info = group_info
            
    @staticmethod
    def find_hdf5_files(directory, minimum_file_size=20e6):
        '''
        Finds all h5 in a directory and returns a tuple of the filenames.
        '''
        files = ()
        for file in os.listdir(directory):
            if file.endswith('.h5'):
                fullname = os.path.join(directory, file)
                if os.stat(fullname).st_size >= minimum_file_size:
                    files += (fullname,)
        return files
        
    def __init__(self, *filenames : str,
                 hdf5_groups=(HdfGroupNamer('DatagramBase/CalculatedData/NavigationSolutionData/slotID0/','dataValidTime','Navigation'),),
                 remove_non_time_stamped_data=True):
        """
        Constructor for SunstoneFileHandler

        Parameters
        ----------
        *filenames : str
            H5 filenames to load.
        hdf5_groups :  tuple of HdfGroupNamer
        remove_non_time_stamped_data : bool. The default is True.
            Remove data where timestamp is zero

        """
        
        self.remove_non_time_stamped_data = remove_non_time_stamped_data
        
        files = ()
        for name in filenames:
            files += (h5py.File(name, 'r'),)
                
        self.groups = {}
        for group_info in hdf5_groups:
            try:
                self.__loadFromPath(files, group_info)
            except:
                raise Exception('Could not load group_info: ' + group_info.name)
        
        for f in files:
            f.close()
        
    def get_signal(self,group_name : str, field_name : str, start_time=None, end_time=None, reset_time_stamp_to_zero=False):
        '''
        Returns a Signal object from a H5 field

        Parameters
        ----------
        group_name : str 
        field_name : str    
        start_time : double or a datatime object.
            The default is None. Removes data before this time.
        end_time : double or a datatime object.
            The default is None. Removes data after this time.
        reset_time_stamp_to_zero : bool.
            The default is False. Resets timestamps in such a way
            that the first sample starts at zero.

        Returns
        -------
        Signal : Signal 
        '''
        
        assert group_name in self.groups, 'group_name ' + group_name + ' does not exist in the loaded data'
        assert field_name in self.groups[group_name].data, 'field_name ' + field_name + ' does not exist in group'
        
        return pp.Signal.create(group_name + '/' + field_name,self.groups[group_name].time,self.groups[group_name].data[field_name],
                                start_time=start_time,end_time=end_time,reset_time_stamp_to_zero=reset_time_stamp_to_zero)
    
    def get_multiple_signals(self,group_name : str, *field_names, start_time=None, end_time=None, reset_time_stamp_to_zero=False): 
        '''
        Returns a tuple of Signal objects from multiple H5 field names.

        Parameters
        ----------
        group_name : str
        *field_names : str.
            A variadic number of H5 field names
        start_time : double or a datatime object.
            The default is None. Removes data before this time.
        end_time : double or a datatime object.
            The default is None. Removes data after this time.
        reset_time_stamp_to_zero : bool.
            The default is False. Resets timestamps in such a way
            that the first sample starts at zero.                

        Returns
        -------
        signals : Signal 

        '''
        signals = tuple()
        for name in field_names:
            signals = signals + (self.get_signal(group_name,name,start_time=start_time,end_time=end_time,reset_time_stamp_to_zero=reset_time_stamp_to_zero),)
        return signals
    
    def __loadFromPath(self, files, group_info : HdfGroupNamer):
        time = np.array([],dtype=np.double)
        data = {}
        
        for f in files:
            try:
                frame = pd.DataFrame(np.array(f[group_info.path]))
                tmp_time = create_time_stamps(frame[group_info.time_field_name],group_info.time_unit)
                indexes =  (tmp_time > 0) if self.remove_non_time_stamped_data else np.full(tmp_time.size,True)
                time = np.append(time,tmp_time[indexes])
                
                for k, v in frame.iteritems():
                    if k in data:
                        data[k] = np.append(data[k],np.array(v)[indexes])
                    else:
                        data[k] = np.array(v)[indexes]
                
            except Exception as e:
                raise Exception('Could not load path',group_info.path,e)
                
        self.groups[group_info.name] = SunstoneFileHandler.GroupData(time, data, group_info)
            
           
def convert_kf_status2time(signal : pp.Signal):
    ii = np.full(signal.size, False)    
    ii[1:] = np.diff(signal.data) > 0
    times = signal.data[ii]    
    times = create_time_stamps(times)
    return pp.Signal(times, np.ones(times.size), signal.name)

def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()

def find_values_based_on_kf_status_timestamps(kf_status_timestamps : pp.Signal, data : pp.Signal,name : str):
    """
    The Sunstone KFStatus field contains information about last input of sensor (timeOfLastInput) and aiding (timeOfLastAiding),
    into the kalman algorithm, e.g. timeOfLastOrientationInput and timeOfLastOrientationAiding.
    This function finds the sensor values from the timeOfLastInput/timeOfLastAiding timestamps.

    Parameters
    ----------
    kf_status_timestamps : pp.Signal
        timeOfLastInput or timeOfLastAiding timestamps
    data : pp.Signal
        The data to find sensor values in, e.g. if timeOfLastRelVelInput sensor values can be taken from DVL water track data
    name : str
        A name to be used in the returned Signal object

    Returns
    -------
    pp.Signal

    """   
    
    aidingTimestamps = pp.convert_kf_status2time(kf_status_timestamps)
    ii = np.logical_and(data.axis[0] <= aidingTimestamps.axis, aidingTimestamps.axis <= data.axis[-1])
    aidingTimestamps.axis = aidingTimestamps.axis[ii]
    aidingTimestamps.data = aidingTimestamps.data[ii]
    
    aidingValues = np.zeros(aidingTimestamps.size)
    for i in range(aidingTimestamps.size):
        aidingValues[i] = data.data[find_nearest_index(data.axis, aidingTimestamps.axis[i])]
    return pp.Signal(aidingTimestamps.axis, aidingValues, name)
    