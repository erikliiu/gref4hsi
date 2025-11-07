# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2021 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************
import time
import decimal


class DotNetTime:
    """Class that holds time as a .NET tick.
    .NET ticks are 100ns ticks since epoch 1/1-0001.
    """
    utc_mask = 1 << 62
    datetimekind_mask = 3 << 62
    unix_epoch_dnt = 621355968000000000
    dnt_per_second = 10000000

    def __init__(self, ticks:int = 0):    
        self.__ticks = ticks & ~DotNetTime.datetimekind_mask  # Reset DateTimeKind bits if set

    @property
    def ticks(self):
        return self.__ticks

    @ticks.setter
    def set_ticks(self, ticks:int):
        self.__ticks = ticks & ~DotNetTime.datetimekind_mask  # Reset DateTimeKind bits if set


    @staticmethod
    def fromUnixEpoch(unix_time_s:float):
        """
        Create a DotNetTime object from a Unix epoch timestamp in seconds.

        Parameters:
            unix_time_s(float)     Unix epoch timestamp in seconds. UTC is expected.

        Returns:
            A DotNetTime in UTC.
        """
        ## TODO can give a 1-tick difference when roundtripping from unix-time->DotNetTime->unix-time, it is the missing rounding in a1 that causes it.
        dnt = int(round(unix_time_s, 7) * DotNetTime.dnt_per_second) + DotNetTime.unix_epoch_dnt
        #a0 = DotNetTime.round_decimal(unix_time_s * DotNetTime.dnt_per_second)
        #a1 = int(unix_time_s * DotNetTime.dnt_per_second)
        #a2 = a1/DotNetTime.dnt_per_second
        #a3 = a2 * DotNetTime.dnt_per_second
        #dnt = int(a3) + DotNetTime.unix_epoch_dnt
        return DotNetTime(dnt)

    def toUnixEpoch(self):
        """Returns the time as seconds since Unix epoch.

        Returns:
            float: Time in seconds since Unix epoch.
        """
        return (self.__ticks - DotNetTime.unix_epoch_dnt) / DotNetTime.dnt_per_second

    @staticmethod
    def round_decimal(value):
        """mimics python's round() function but with half-point up rounding"""
        return int(decimal.Decimal.from_float(value).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

    @staticmethod
    def nowUtc():
        """Returns a DotNetTime object containing UTC time now as value.
        """
        return DotNetTime.fromUnixEpoch(time.time())