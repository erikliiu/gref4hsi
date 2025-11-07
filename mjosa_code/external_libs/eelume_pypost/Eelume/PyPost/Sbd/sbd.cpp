// encoding: utf-8
// ***************************************************************************
// Copyright (C) 2022 Eelume AS - All Rights Reserved
// 
// Unauthorized copying of this file, via any medium is strictly prohibited.
// Proprietary and confidential.
// ***************************************************************************

#include "sbd.h"
#include <assert.h>
#include <stdio.h>

int sbd_get_buf(const char* filename, char*& buf, int& buf_size)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return -1;

    fseek(f, 0, SEEK_END);
    buf_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (char*)malloc(buf_size);
    if (!buf)
        return -1;

    {
        size_t read = fread(buf, 1, buf_size, f);
        if (read != buf_size)
            return -1;
    }
    fclose(f);

    return 0;
}

const sbd_entry_header_t* sbd_get_next_header(const sbd_entry_header_t* header)
{
    return (const sbd_entry_header_t*)((const char*)(header + 1) + header->entry_size);
}

bool sbd_valid_header(const sbd_entry_header_t* header, const char* end)
{
    return ((const char*)header < end) && ((const char*)(header + 1) + header->entry_size < end);
}

int sbd_get_remaining_headers(sbd_entry_header_t::entry_type_t type, const char* buf, int buf_size)
{
    int ret = 0;

    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; sbd_valid_header(header, (const char*)buf + buf_size); header = sbd_get_next_header(header))
        if (header->entry_type == type)
            ++ret;

    return ret;
}

int sbd_get_bath(const char* buf, int buf_size, sbd_bath_entry_t*& entries, int& n_entries)
{
    n_entries = sbd_get_remaining_headers(sbd_entry_header_t::WBMS_BATH, buf, buf_size);
    entries = (sbd_bath_entry_t*)malloc(sizeof(*entries) * n_entries);

    int bath_index = 0;
    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; bath_index < n_entries; header = sbd_get_next_header(header))
    {
        //printf("header: type: 0x%02x relative_time: %d, absolute_time: %d %d, size: %d\n", header->entry_type, header->relative_time, header->absolute_time.tv_sec, header->absolute_time.tv_usec, header->entry_size);

        if (header->entry_type == sbd_entry_header_t::WBMS_BATH)
        {
            assert(header->entry_size == 10352);

            bath_data_packet_t* bath = (bath_data_packet_t*)(header + 1);
            assert(bath->size() == header->entry_size);
            assert(bath->size() == sizeof(*bath));
            assert(bath->header.preamble == 0xdeadbeef);
            assert(bath->sub_header.N == 512);
            assert(bath->sub_header.sample_rate == 78125.f);
            //printf("ping_number: %d, freq: %f, tx_angle: %f, swath_open: %f, time: %f\n", bath->sub_header.ping_number, bath->sub_header.tx_freq, bath->sub_header.tx_angle, bath->sub_header.swath_open, bath->sub_header.time);

            entries[bath_index].timestamp = bath->sub_header.time;

            for (int beam_index = 0; beam_index < bath->sub_header.N; ++beam_index)
            {
                sbd_bath_entry_t* bath_entry = entries + bath_index;
                //printf("sample_number: %u %f %u %u %f %u %u %u\n", bath->dp[beam_index].sample_number, bath->dp[beam_index].angle, bath->dp[beam_index].upper_gate, bath->dp[beam_index].lower_gate, bath->dp[beam_index].intensity, bath->dp[beam_index].flags, bath->dp[beam_index].quality_flags, bath->dp[beam_index].quality_val);

                bath_entry->range[beam_index] = (bath->dp[beam_index].sample_number * bath->sub_header.snd_velocity) / (2 * bath->sub_header.sample_rate);
                bath_entry->angle[beam_index] = bath->dp[beam_index].angle;
            }

            ++bath_index;
        }
    }

    return 0;
}

int sbd_get_lat_long(const char* buf, int buf_size, sbd_lat_long_entry_t*& entries, int& n_entries)
{
    n_entries = sbd_get_remaining_headers(sbd_entry_header_t::NMEA_EIPOS, buf, buf_size);
    entries = (sbd_lat_long_entry_t*)malloc(sizeof(*entries) * n_entries);

    int index = 0;
    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; index < n_entries; header = sbd_get_next_header(header))
    {
        if (header->entry_type == sbd_entry_header_t::NMEA_EIPOS)
        {
            const char* nmea = (const char*)(header + 1) + 20;
            //printf("nmea: %s", nmea);

            uint32_t len;
            long long unsigned timestamp;
            double timeUTC;
            double longitude;
            double latitude;
            char north;
            char east;
            int sscanf_ret = sscanf(nmea,"$EIPOS,%u,%lf,%llu,%lf,%c,%lf,%c*",&len,&timeUTC,&timestamp,&latitude,&north, &longitude,&east);

            assert(sscanf_ret == 7);

            entries[index].timestamp = (double)timestamp / 1e3;
            entries[index].latitude = latitude;
            entries[index].longitude = longitude;

            ++index;
        }
    }

    return 0;
}

int sbd_get_roll_pitch(const char* buf, int buf_size, sbd_roll_pitch_entry_t*& entries, int& n_entries)
{
    n_entries = sbd_get_remaining_headers(sbd_entry_header_t::NMEA_EIORI, buf, buf_size);
    entries = (sbd_roll_pitch_entry_t*)malloc(sizeof(*entries) * n_entries);

    int index = 0;
    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; index < n_entries; header = sbd_get_next_header(header))
    {
        if (header->entry_type == sbd_entry_header_t::NMEA_EIORI)
        {
            const char* nmea = (const char*)(header + 1) + 16;

            uint32_t len;
            long long unsigned timestamp;
            double timeUTC;
            double roll;
            double pitch;
            int sscanf_ret = sscanf(nmea,"$EIORI,%u,%lf,%llu,%lf,%lf*",&len,&timeUTC,&timestamp,&roll,&pitch);

            assert(sscanf_ret == 5);

            entries[index].timestamp = (double)timestamp / 1e3;
            entries[index].roll = roll;
            entries[index].pitch = pitch;

            ++index;
        }
    }

    return 0;
}

int sbd_get_heading(const char* buf, int buf_size, sbd_heading_entry_t*& entries, int& n_entries)
{
    n_entries = sbd_get_remaining_headers(sbd_entry_header_t::NMEA_EIHEA, buf, buf_size);
    entries = (sbd_heading_entry_t*)malloc(sizeof(*entries) * n_entries);

    int index = 0;
    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; index < n_entries; header = sbd_get_next_header(header))
    {
        if (header->entry_type == sbd_entry_header_t::NMEA_EIHEA)
        {
            const char* nmea = (const char*)(header + 1) + 20;

            uint32_t len;
            long long unsigned timestamp;
            double timeUTC;
            double heading;
            int sscanf_ret = sscanf(nmea,"$EIHEA,%u,%lf,%llu,%lf*",&len,&timeUTC,&timestamp,&heading);

            assert(sscanf_ret == 4);

            entries[index].timestamp = (double)timestamp / 1e3;
            entries[index].heading = heading;

            ++index;
        }
    }

    return 0;
}

int sbd_get_depth_altitude(const char* buf, int buf_size, sbd_depth_altitude_entry_t*& entries, int& n_entries)
{
    n_entries = sbd_get_remaining_headers(sbd_entry_header_t::NMEA_EIDEP, buf, buf_size);
    entries = (sbd_depth_altitude_entry_t*)malloc(sizeof(*entries) * n_entries);

    int index = 0;
    for (const sbd_entry_header_t* header = (sbd_entry_header_t*)buf; index < n_entries; header = sbd_get_next_header(header))
    {
        if (header->entry_type == sbd_entry_header_t::NMEA_EIDEP)
        {
            const char* nmea = (const char*)(header + 1) + 16;

            uint32_t len;
            long long unsigned timestamp;
            double timeUTC;
            double depth;
            double altitude;
            int sscanf_ret = sscanf(nmea,"$EIDEP,%u,%lf,%llu,%lf,m,%lf,m*",&len,&timeUTC,&timestamp,&depth,&altitude);

            assert(sscanf_ret == 5);

            entries[index].timestamp = (double)timestamp / 1e3;
            entries[index].depth = depth;
            entries[index].altitude = altitude;

            ++index;
        }
    }

    return 0;
}
