# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************
import array
import numpy as np
import sqlite3
from enum import IntEnum
from Eelume.PyPost.DotNetTime import DotNetTime
from Eelume.PyPost.Signal import Signal
import os


class DataType(IntEnum):  #
    Unknown = (0,)  # Unknown data type.
    Bool = (1,)  # Data type is a boolean.
    Int8 = (2,)  # Data type is a signed 8bit integer.
    UInt8 = (3,)  # Data type is an unsigned 8bit integer.
    Int16 = (4,)  # Data type is a signed 16bit integer.
    UInt16 = (5,)  # Data type is an unsigned 16bit integer.
    Int32 = (6,)  # Data type is a signed 32bit integer.
    UInt32 = (7,)  # Data type is an unsigned 32bit integer.
    Int64 = (8,)  # Data type is a signed 64bit integer.
    UInt64 = (9,)  # Data type is an unsigned 64bit integer.
    Float = (10,)  # Data type is a floating point number with single precision.
    Double = (11,)  # Data type is a floating point number with double precision.
    ByteArray = (12,)  # Data type is a byte array.
    String = (13,)  # Data type is a string.
    Matrix = (14,)  # Data type is a Eigen matrix.
    Vector = 15  # Data type is a Eigen vector.


class DatabaseHandler(object):
    def __init__(self, filename: str, erase_non_ntp_timestamed_data=True):
        """
        Constructor for DatabaseHandler which connects to a sqlite database and
        finds available data base entries (paths)

        Parameters
        ----------
        filename : str
            Database filename.
        erase_non_ntp_timestamed_data  : bool
            The default is True. The database might start the logging
            before NTP server is up and running. Hence, a large step in the timestamp
            is seen. Set this parameter to true erase data that was logged before
            NTP server was started.

        """

        if not os.path.exists(filename):
            raise Exception("Filename does not exist: " + filename)

        self.__cursor = sqlite3.connect(
            filename
        ).cursor()  # ? creates a cursor to the database, which allows us to execute SQL commands
        self.__createIndexed()
        self.__init_all_paths()
        self.erase_non_ntp_timestamed_data = erase_non_ntp_timestamed_data

    def get_signal(
        self,
        # ? eg pathname = "navigation/robot_position/lat_deg"
        path_name: str,
        start_time=None,
        end_time=None,
        reset_time_stamp_to_zero=False,
    ) -> Signal:
        """
        Returns a Signal object from a database entry

        Parameters
        ----------
        path_name : str.
            The path name of the database entry.
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
        """
        # ? assert is used to make sure that the right type of argument is passed to the function -> which here is string
        assert type(path_name) is str

        # Retrieve the axis(timeaxis) and data(val at that time) for the given path_name(eg /sensors/altimeter/altitude) from the database
        axis, data = self.__selectDataEntry(path_name)

        return Signal.create(
            path_name, axis, data, start_time, end_time, reset_time_stamp_to_zero
        )

    def get_multiple_signals(
        # use when u want to get signals for a specific thing, like motion in x, y, z, then u have 3 signals, but for the same "category"
        self,  # ? The self parameter is automatically passed when calling a method inside a class. That’s why you don’t see it explicitly written when calling the function.
        *path_names: str,
        start_time=None,
        end_time=None,
        reset_time_stamp_to_zero=False
    ) -> tuple:
        """
        Returns a tuple of Signal objects from multiple database entries.

        Parameters
        ----------
        *path_names : str.
            A variadic number of database entry names
        start_time : double or a datatime object.
            The default is None. Removes data before this time.
        end_time : double or a datatime object.
            The default is None. Removes data after this time.
        reset_time_stamp_to_zero : bool.
            The default is False. Resets timestamps in such a way
            that the first sample starts at zero.

        Returns
        -------
        signals : tuple(Signal)

        """

        signals = tuple()
        # ? loop through each of the path_names and get the signal for each of them
        # ? eg name = "navigation/robot_position/lat_deg"
        for name in path_names:
            # retrieves the signal for the given path (eg /sensors/altimeter/altitude) and then makes a new tuple (creates a new one since it is immutable)
            signals = signals + (
                self.get_signal(
                    name,
                    start_time=start_time,
                    end_time=end_time,
                    reset_time_stamp_to_zero=reset_time_stamp_to_zero,
                ),
            )
        # returns a tuple of signals, one signal for each path_name (eg /sensors/altimeter/altitude)
        return signals

    def get_entry_paths(self):
        """
        Returns the availabe database entry paths

        Returns
        -------
        list
            A list of available entry paths. Each list element contains a
            dictionary of 'rowid', 'path', 'datatype' and 'shape'.

        """
        return list(self.__paths.values())

    def search_for_str_in_paths(self, searchStr: str):
        """
        Searches all available database entry paths and returns
        a list of paths that contains the searchStr

        Parameters
        ----------
        searchStr : str

        Returns
        -------
        res : list of str

        """
        res = []
        for path in self.get_entry_paths():
            if path["path"].find(searchStr) != -1:
                res.append(path["path"])
        return res

    def __del__(self):
        self.__cursor.close()

    def __init_all_paths(self):
        ## Check  if column 'shape' exists, it was added in database format 3.1

        # ? sql is a string that contains the SQL query (command written in SQL that is used to ask a database for information) to be executed
        # ? COUNT(*) -> counts how many times the column shape appears in the table DataTable_Paths
        # ? gives the result (either 0 or 1) a custom name 'shape_column_exists'
        sql = "SELECT COUNT(*) AS shape_column_exists FROM pragma_table_info('DataTable_Paths') WHERE name ='shape'"

        # ? cursor is like a tool that talks to the database and runs SQL commands (query)
        # ? cursor.execute(sql) runs the sql query stored in the variable sql. the result stays inside the cuyrsor, waiting to be fetched
        # ? fetchall() retrieves all the rows from the last executed query. it returns a list of tuples eg. [(1,)] (if there is one row) or [(0,)] (if column shape does not exist)
        # ? res is a list of tuples, where each tuple contains one element (either 0 or 1). idealy should contain [(1,)] which indicate that column named "shape" exists
        res = self.__cursor.execute(sql).fetchall()

        # ? shape_col_exists is a boolean variable that is set to False, changed later if the column shape exists
        shape_col_exists = False

        # ? if the length of the result is not 1, then raise an exception. this is because we should only have one column in the sql table named 'shape'
        if len(res) != 1:
            raise Exception(
                "Unexpected SQLite result, expected 1 row, got {} rows".format(len(res))
            )

        # ? for each tuple in the list res, check if the first element of the tuple is 1. if it is, then set shape_col_exists to True
        # ? info = (1,) or info = (0,) where info[0] is either 1 or 0
        for info in res:
            if info[0] == 1:
                shape_col_exists = True

        if shape_col_exists:
            # ? SELECT → Retrieves specific columns from the table.
            # ? rowid, Path, datatype, shape → These are the columns we want.
            # ? FROM DataTable_Paths → We are getting data from this table.
            # ? ORDER BY Path → Sorts the results based on the Path column.
            # ? SELECT → Retrieves specific columns from the table.
            # ? rowid, Path, datatype, shape → These are the columns we want.
            # ? FROM DataTable_Paths → We are getting data from this table.
            # ? ORDER BY Path → Sorts the results based on the Path column.
            sql = (
                "SELECT rowid, Path, datatype, shape FROM DataTable_Paths ORDER BY Path"
            )
        else:
            ## 'shape' column do not exist then shape for all variables is (1,1) (a scalar)
            # ? okey to drop shape, since this is a variable that describes the structure of the data. say we are looking at roll, then shape would be (1,1) since it is a scalar, while for thruster forces it could be a vector (3,1) for x, y, z
            sql = "SELECT rowid, Path, datatype FROM DataTable_Paths ORDER BY Path"

        # ? eg datainfos = [(1, "path1", 6, (1, 1)), (2, "path2", 10, (2, 2))]
        # ? datainfoes variable contains metdadata about the data stored in the database, such as the structure and type of the data, but it does not contain the actuall data values themself, it provides information on how to retrieve and interpret the data
        datainfos = self.__cursor.execute(sql).fetchall()

        # ? a dictionary that stores all metadata for different data paths in the database (key = path, value = metadata). contains every single row found in the "DataTable_Paths" table (see DQLite) ie a dict with 6601 items
        all_paths_dict = dict()

        # ? loop thru datainfos (a list of metadata entires). each datainfo is a row found in "DataTable_Paths" as seen in SQLite
        for datainfo in datainfos:

            # ? note we store all info for that row except ticks_utc_dotnet_epoch -> we dont need the time stamp, only storing metadata (data about data)
            # ? each row shall have a key that contain information about that row, which of course is the row id, path, datatype and shape if it exists.
            # ? array.array is like a list, but more restricted (only one type of data) and thus is more memory efficient
            path_dict = dict()
            path_dict["rowid"] = datainfo[0]
            path_dict["path"] = datainfo[1]
            path_dict["datatype"] = datainfo[2]
            if shape_col_exists:
                # ? shape describes the structure of the data, and in sql it is stored as BLOB, we want to convert int back into list of int.
                # ? array.array("i", datainfo[3]) converts the binary into an integer array (is a special tuype of list that stores only one type of data).
                # Shape is stored in db as BLOB as a byte-array of int32 values.
                path_dict["shape"] = array.array(
                    "i",
                    datainfo[3],
                )
            else:
                path_dict["shape"] = array.array(
                    "i", [1, 1]
                )  # Shape not found, then the shape is always (1,1) (a scalar)

            # ? adds that one row with 4 different keys/val as a key to the huge dict containg all the rows in the log file
            all_paths_dict[path_dict["path"]] = path_dict

        """
        eg. paths:

        {
            "/sensors/temp": {
                "rowid": 5347,
                "path": "/sensors/temp",
                "datatype": 1,
                "shape": array.array("i", [3])  # Converted from BLOB
            },
            "/guidance/DUNE_has_control": {
                "rowid": 5348,
                "path": "/guidance/DUNE_has_control",
                "datatype": 1,
                "shape": array.array("i", [1,1])  # Default (1,1) if shape not found
            }
        }

        """
        self.__paths = all_paths_dict

    def __createIndexed(self):
        create_index_sql = "CREATE INDEX IF NOT EXISTS %s ON %s(%s)"
        self.__cursor.execute(
            create_index_sql % ("indx_path", "DataTable_Paths", "Path")
        )
        self.__cursor.execute(
            create_index_sql % ("indx_dbl_path_rowid", "DataTable_Double", "Path_Rowid")
        )

    def __copyEntryToArray(self, rowid: int, sql: str, dtype):
        """
        Runs an SQL query using the cursor, filtering by path_rowid.

        - Executes `sql`, which is the command given before.
        - Uses `rowid` to filter results (only rows where path_rowid matches).
        - `fetchall()` collects all matching rows and stores them in `res`.
        - `res` is a list of tuples (each tuple is a row from the database).

        Example:
        --------
        If `sql = "SELECT value_int32 FROM DataTable_Int32 WHERE path_rowid=?"`
        and `rowid = 4474`, then:

            res = self.__cursor.execute(sql, (4474,)).fetchall()

        Might return:

            [(10,), (20,), (30,)]  # List of tuples (one per row)
        """

        res = self.__cursor.execute(sql, (rowid,)).fetchall()
        nSamples = len(res)
        # ? np.empty are just random leftover val from memory, so we need to fill them with the actual data from the database
        axis = np.empty(nSamples, dtype=np.double)  # time axis
        data = np.empty(nSamples, dtype=dtype)  # data axis

        for i in range(nSamples):
            # convert time from UTC (sql database) to unix epoch. eg ticks_utc_dotnet_epoch = 637497024000000000 -> unix_epoch = 1635475200.0
            axis[i] = DotNetTime(res[i][0]).toUnixEpoch()

            # takes out data found in "value_int32" in sql -> elem [1], the first elem [0] is the time stamp
            data[i] = res[i][1]

        # returns axis (time) and data (values) for the given path (/sensors/altimeter/altitude) which has a unique corresponding rowid
        return axis, data

    def __read_array_double_signal(
        self, db_cursor: sqlite3.Cursor, rowid: int, shape: array.array
    ):
        data = None
        axis = None
        sql = "SELECT arr.ticks_utc_dotnet_epoch, arr.data FROM DataTable_Array_Double AS arr WHERE path_rowid=? ORDER BY arr.ticks_utc_dotnet_epoch"
        res = db_cursor.execute(sql, (rowid,)).fetchall()

        shape.append(
            len(res)
        )  # Extend shape of resulting array with the number of rows in the signal
        idx = 0
        for row in res:
            if data is None:
                data = np.ndarray(shape)
                axis = np.empty(len(res))
            tmp = np.frombuffer(row[1], dtype=np.double)
            tmp = np.reshape(tmp, newshape=shape[:-1])
            if (
                tmp.shape[0] > 1 and tmp.shape[1] > 2
            ):  ## TODO this works for vectors and matrices, but I'm not sure it will work for 3-dim arrays.
                tmp = (
                    tmp.transpose()
                )  # Array data are stored column-major in database, but read row-major in Numpy
            data[..., idx] = tmp
            axis[idx] = DotNetTime(row[0]).toUnixEpoch()
            idx += 1
        return axis, data

    def __eraseNonNTPTimestamedData(self, timeAxis, data):
        if len(timeAxis) > 1:
            timeStepLimit = 3600 * 24
            tdiff = np.diff(timeAxis)
            index = np.argmax(tdiff > timeStepLimit)
            if index > 0:
                timeAxis = timeAxis[index + 1 :]
                data = data[..., index + 1 :]
            else:
                if (
                    tdiff[0] > timeStepLimit
                ):  # argmax return zero also if tdiff[0] > timeStepLimit, hence extra check for tdiff[0]
                    timeAxis = timeAxis[1:]
                    data = data[..., 1:]

        return timeAxis, data

    def __selectDataEntry(self, path: str):
        # ? check if the path is in the dictionary of paths (each row found in the "DataTable_Paths" table is a key in this dict)
        if path in self.__paths:
            # ? if the path is in the dictionary, then retrieve all value associated with that key
            # ? datainfo variable -> contains metadata about the data stored at that row, such as rowid, path, datatype and shape
            # ? eg path = "/navigation/robot_position/lat_deg"

            """
            eg

            self.__paths = {
                "path1": {"rowid": 1, "path": "path1", "datatype": DataType.Int32, "shape": [1, 1]},
                "path2": {"rowid": 2, "path": "path2", "datatype": DataType.Float, "shape": [2, 2]},
            }

            if path = "path1", then datainfo = {"rowid": 1, "path": "path1", "datatype": DataType.Int32, "shape": [1, 1]}
            """
            datainfo = self.__paths[path]
        else:
            raise Exception("Path not found: " + path)

        """
        SQL Query Explanation (sqlInt32):
        ---------------------------------
        This string variable holds our SQL command:

            SELECT Int32.ticks_utc_dotnet_epoch, Int32.value_int32
            FROM DataTable_Int32 AS Int32
            WHERE path_rowid=?
            ORDER BY Int32.ticks_utc_dotnet_epoch

        - SELECT ...: Chooses the columns 'ticks_utc_dotnet_epoch' (time stamp) and 'value_int32' (integer value).
        - FROM DataTable_Int32 AS Int32: Uses the 'DataTable_Int32' table, giving it an alias 'Int32' for shorter references.
        - WHERE path_rowid=? : The '?' is a placeholder we fill with a specific path_rowid, filtering rows by that ID.
        - ORDER BY Int32.ticks_utc_dotnet_epoch: Sorts the returned rows chronologically by their time stamp.
        """

        sqlInt32 = "SELECT Int32.ticks_utc_dotnet_epoch, Int32.value_int32 FROM DataTable_Int32 AS Int32 WHERE path_rowid=? ORDER BY Int32.ticks_utc_dotnet_epoch"
        sqlInt64 = "SELECT Int64.ticks_utc_dotnet_epoch, Int64.value_int64 FROM DataTable_Int64 AS Int64 WHERE path_rowid=? ORDER BY Int64.ticks_utc_dotnet_epoch"

        rowid = datainfo["rowid"]
        path = datainfo["path"]
        datatype = datainfo["datatype"]
        shape = datainfo["shape"]

        if datatype == DataType.Bool:
            axis, data = self.__copyEntryToArray(
                rowid=rowid,
                sql="SELECT Bool.ticks_utc_dotnet_epoch, Bool.value_bool FROM DataTable_Bool AS Bool WHERE path_rowid=? ORDER BY Bool.ticks_utc_dotnet_epoch",
                dtype=bool,
            )
        elif datatype == DataType.Int8:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.int8
            )
        elif datatype == DataType.UInt8:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.uint8
            )
        elif datatype == DataType.Int16:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.int16
            )
        elif datatype == DataType.UInt16:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.uint16
            )
        elif datatype == DataType.Int32:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.int32
            )
        elif datatype == DataType.UInt32:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt32, dtype=np.uint32
            )
        elif datatype == DataType.Int64:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt64, dtype=np.int64
            )
        elif datatype == DataType.UInt64:
            axis, data = self.__copyEntryToArray(
                rowid=rowid, sql=sqlInt64, dtype=np.uint64
            )
        elif datatype == DataType.Float:
            axis, data = self.__copyEntryToArray(
                rowid=rowid,
                sql="SELECT Dbl.ticks_utc_dotnet_epoch, Dbl.value_double FROM DataTable_Double AS Dbl WHERE path_rowid=? ORDER BY Dbl.ticks_utc_dotnet_epoch",
                dtype=np.float32,
            )
        elif datatype == DataType.Double:
            axis, data = self.__copyEntryToArray(
                rowid=rowid,
                sql="SELECT Dbl.ticks_utc_dotnet_epoch, Dbl.value_double FROM DataTable_Double AS Dbl WHERE path_rowid=? ORDER BY Dbl.ticks_utc_dotnet_epoch",
                dtype=np.double,
            )
        elif datatype == DataType.String:
            axis, data = self.__copyEntryToArray(
                rowid=rowid,
                sql="SELECT Str.ticks_utc_dotnet_epoch, Str.value FROM DataTable_String AS Str WHERE path_rowid=? ORDER BY Str.ticks_utc_dotnet_epoch",
                dtype=object,
            )
        elif datatype == DataType.Vector:
            axis, data = self.__read_array_double_signal(self.__cursor, rowid, shape)

            if axis.size == 1:
                data = np.reshape(data, (data.shape[0], 1))
            else:
                data = np.squeeze(data)

        elif datatype == DataType.Matrix:
            axis, data = self.__read_array_double_signal(self.__cursor, rowid, shape)
        else:
            raise Exception("DataType is not implemented: " + str(datainfo[2]))

        if self.erase_non_ntp_timestamed_data:
            axis, data = self.__eraseNonNTPTimestamedData(axis, data)

        return axis, data


if __name__ == "__main__":
    pass
