// encoding: utf-8
// ***************************************************************************
// Copyright (C) 2022 Eelume AS - All Rights Reserved
// 
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
// ***************************************************************************

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "sbd.h"

namespace py = pybind11;

struct pysbd
{
    char* buf = nullptr;
    int buf_size = 0;

    py::module_ pp = py::module_::import("Eelume.PyPost");
    py::object create_signal = pp.attr("Signal").attr("create");

    struct error : public std::runtime_error
    {
        error(const char* s) : std::runtime_error(s){}
        error(const std::string& s) : std::runtime_error(s){}
    };

    pysbd(const char* filename)
    {
        int rc = sbd_get_buf(filename, buf, buf_size);
        if (rc != 0)
            throw error("no such file: " + std::string(filename));
    }

    // map data from fields in an array of c++ structs into a numpy array
    template <typename ENTRY_T, typename FIELD_T>
    static inline std::enable_if_t<std::is_fundamental_v<FIELD_T>,
    py::object> np_array_from_fields(ENTRY_T* entries, int n_entries, FIELD_T ENTRY_T::*field)
    {
        return py::array_t<FIELD_T>
        {
            std::array<int,1>{n_entries}, // shape
            std::array<int,1>{sizeof(ENTRY_T)}, // stride
            (FIELD_T*)((char*)entries + ((char*)&((ENTRY_T*)nullptr->*field) - (char*)nullptr)), // data + offset of data field
            py::capsule(entries, [](void *b) {}) // destructor
        };
    }

    // for bathymetry data, etc. we generate 2d numpy arrays
    template <typename ENTRY_T, typename FIELD_T>
    static inline std::enable_if_t<std::is_array_v<FIELD_T>,
    py::object> np_array_from_fields(ENTRY_T* entries, int n_entries, FIELD_T ENTRY_T::*field)
    {
        using subentry_t = std::remove_reference_t<decltype(FIELD_T{}[0])>; // get type of array entries

        return py::array_t<subentry_t, py::array::c_style>
        {
            std::array<int,2>{sizeof(FIELD_T) / sizeof(subentry_t), n_entries}, // shape
            std::array<int,2>{sizeof(subentry_t), sizeof(ENTRY_T)}, // stride
            (subentry_t*)((char*)entries + ((char*)&((ENTRY_T*)nullptr->*field) - (char*)nullptr)), // data + offset of data field
            py::capsule(entries, [](void *b) {}) // destructor
        };
    }

    std::array<py::object, 2> get_bath(py::object start_time, py::object end_time, py::object reset_time_stamp_to_zero)
    {
        sbd_bath_entry_t* entries;
        int n_entries;
        int rc = sbd_get_bath(buf, buf_size, entries, n_entries);
        if (rc != 0)
            throw error("sbd parse error");

        auto time = np_array_from_fields(entries, n_entries, &sbd_bath_entry_t::timestamp);
        auto range = np_array_from_fields(entries, n_entries, &sbd_bath_entry_t::range);
        auto angle = np_array_from_fields(entries, n_entries, &sbd_bath_entry_t::angle);

        return {
            create_signal("range", time, range, start_time, end_time, reset_time_stamp_to_zero),
            create_signal("angle", time, angle, start_time, end_time, reset_time_stamp_to_zero),
        };
    }

    std::array<py::object, 2> get_lat_long(py::object start_time, py::object end_time, py::object reset_time_stamp_to_zero)
    {
        sbd_lat_long_entry_t* entries;
        int n_entries;
        int rc = sbd_get_lat_long(buf, buf_size, entries, n_entries);
        if (rc != 0)
            throw error("sbd parse error");

        auto time = np_array_from_fields(entries, n_entries, &sbd_lat_long_entry_t::timestamp);
        auto latitude = np_array_from_fields(entries, n_entries, &sbd_lat_long_entry_t::latitude);
        auto longitude = np_array_from_fields(entries, n_entries, &sbd_lat_long_entry_t::longitude);

        return {
            create_signal("latitude", time, latitude, start_time, end_time, reset_time_stamp_to_zero),
            create_signal("longitude", time, longitude, start_time, end_time, reset_time_stamp_to_zero),
        };
    }

    std::array<py::object, 2> get_roll_pitch(py::object start_time, py::object end_time, py::object reset_time_stamp_to_zero)
    {
        sbd_roll_pitch_entry_t* entries;
        int n_entries;
        int rc = sbd_get_roll_pitch(buf, buf_size, entries, n_entries);
        if (rc != 0)
            throw error("sbd parse error");

        auto time = np_array_from_fields(entries, n_entries, &sbd_roll_pitch_entry_t::timestamp);
        auto roll = np_array_from_fields(entries, n_entries, &sbd_roll_pitch_entry_t::roll);
        auto pitch = np_array_from_fields(entries, n_entries, &sbd_roll_pitch_entry_t::pitch);
    
        return {
            create_signal("roll", time, roll, start_time, end_time, reset_time_stamp_to_zero),
            create_signal("pitch", time, pitch, start_time, end_time, reset_time_stamp_to_zero),
        };
    }

    py::object get_heading(py::object start_time, py::object end_time, py::object reset_time_stamp_to_zero)
    {
        sbd_heading_entry_t* entries;
        int n_entries;
        int rc = sbd_get_heading(buf, buf_size, entries, n_entries);
        if (rc != 0)
            throw error("sbd parse error");

        auto time = np_array_from_fields(entries, n_entries, &sbd_heading_entry_t::timestamp);
        auto heading = np_array_from_fields(entries, n_entries, &sbd_heading_entry_t::heading);

        return create_signal("heading", time, heading, start_time, end_time, reset_time_stamp_to_zero);

        //return p::make_tuple(time, heading);
    }

    std::array<py::object, 2> get_depth_altitude(py::object start_time, py::object end_time, py::object reset_time_stamp_to_zero)
    {
        sbd_depth_altitude_entry_t* entries;
        int n_entries;
        int rc = sbd_get_depth_altitude(buf, buf_size, entries, n_entries);
        if (rc != 0)
            throw error("sbd parse error");

        auto time = np_array_from_fields(entries, n_entries, &sbd_depth_altitude_entry_t::timestamp);
        auto depth = np_array_from_fields(entries, n_entries, &sbd_depth_altitude_entry_t::depth);
        auto altitude = np_array_from_fields(entries, n_entries, &sbd_depth_altitude_entry_t::altitude);

        return {
            create_signal("depth", time, depth, start_time, end_time, reset_time_stamp_to_zero),
            create_signal("altitude", time, altitude, start_time, end_time, reset_time_stamp_to_zero),
        };
    }
};

#define DEFAULT_ARGUMENTS py::arg("start_time")=py::none(), py::arg("end_time")=py::none(), py::arg("reset_time_stamp_to_zero")=py::bool_(false)

PYBIND11_MODULE(pysbd, m)
{
    py::class_<pysbd>(m, "pysbd")
        .def(py::init<const char*>())
        .def("get_bath", &pysbd::get_bath, DEFAULT_ARGUMENTS)
        .def("get_lat_long", &pysbd::get_lat_long, DEFAULT_ARGUMENTS)
        .def("get_roll_pitch", &pysbd::get_roll_pitch, DEFAULT_ARGUMENTS)
        .def("get_heading", &pysbd::get_heading, DEFAULT_ARGUMENTS)
        .def("get_depth_altitude", &pysbd::get_depth_altitude, DEFAULT_ARGUMENTS);
}
