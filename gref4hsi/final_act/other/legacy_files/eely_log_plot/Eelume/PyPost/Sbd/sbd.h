// encoding: utf-8
// ***************************************************************************
// Copyright (C) 2022 Eelume AS - All Rights Reserved
// 
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
// ***************************************************************************

#include <stdint.h>
#include <time.h>
#include <functional>

#include "bath.h"

static constexpr int sbd_n_beams = sizeof(bath_data_packet_t::dp) / sizeof(bath_data_packet_t::detection_point_t); // TODO: dont assume number of beams

struct sbd_entry_header_t
{
    enum entry_type_t : char
    {
        NMEA_EIHEA = 2,
        NMEA_EIORI = 3,
        NMEA_EIDEP = 4,
        NMEA_EIPOS = 8,
        WBMS_BATH = 9,
        HEADER = 21,
    } entry_type;

    char dont_care[3];
    int32_t relative_time;

    struct
    {
        int32_t tv_sec;
        int32_t tv_usec;
    } absolute_time;

    uint32_t entry_size;
};
static_assert(sizeof(sbd_entry_header_t) == 20);


struct sbd_heading_entry_t
{
    double timestamp; // seconds
    float heading;
};

struct sbd_roll_pitch_entry_t
{
    double timestamp;
    float roll;
    float pitch;
};

struct sbd_depth_altitude_entry_t
{
    double timestamp;
    float depth;
    float altitude;
};

struct sbd_lat_long_entry_t
{
    double timestamp;
    float latitude;
    float longitude;
};

struct sbd_bath_entry_t
{
    double timestamp;
    float range[sbd_n_beams];
    float angle[sizeof(range) / sizeof(*range)];
};


int sbd_get_buf(const char* filename, char*& buf, int& buf_size);

int sbd_get_bath(const char* buf, int buf_size, sbd_bath_entry_t*& entries, int& n_entries);

int sbd_get_lat_long(const char* buf, int buf_size, sbd_lat_long_entry_t*& entries, int& n_entries);

int sbd_get_roll_pitch(const char* buf, int buf_size, sbd_roll_pitch_entry_t*& entries, int& n_entries);

int sbd_get_heading(const char* buf, int buf_size, sbd_heading_entry_t*& entries, int& n_entries);

int sbd_get_depth_altitude(const char* buf, int buf_size, sbd_depth_altitude_entry_t*& entries, int& n_entries);
