#!/usr/bin/env python
"""
Scripts for evalution of WNUT 2016 geotagging shared task
"""

import argparse
import ujson as json
import math

EARTH_RADIUS = 6372.8


def _calc_dist_radian(pLat, pLon, lat, lon):
    """
    Calculate the Great Circle Distance between two points on earth
    http://en.wikipedia.org/wiki/Great-circle_distance
    """
    cos_pLat = math.cos(pLat)
    sin_pLat = math.sin(pLat)
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    long_diff = pLon - lon
    cos_long_diff = math.cos(long_diff)
    sin_long_diff = math.sin(long_diff)
    numerator = math.sqrt(math.pow(cos_lat * sin_long_diff, 2) +
                          math.pow(cos_pLat * sin_lat - sin_pLat * cos_lat * cos_long_diff, 2))
    denominator = sin_pLat * sin_lat + cos_pLat * cos_lat * cos_long_diff
    radian = math.atan2(numerator, denominator)
    return radian * EARTH_RADIUS


def _degree_radian(degree):
    return (degree * math.pi) / 180


def calc_dist_degree(pLat, pLon, lat, lon):
    pLat = _degree_radian(pLat)
    pLon = _degree_radian(pLon)
    lat = _degree_radian(lat)
    lon = _degree_radian(lon)
    return _calc_dist_radian(pLat, pLon, lat, lon)


def evaluate_submission(output_file, oracle_file, submission_type):
    output_data = etl_data(output_file, submission_type)
    oracle_data = etl_data(oracle_file, submission_type)
    assert len(output_data) == len(oracle_data)
    accuracy, median_error, mean_error = 0.0, 0.0, 0.0
    right, wrong = 0, 0
    error_distance_list = []
    for output, oracle in zip(output_data, oracle_data):
        assert output[0] == oracle[0]
        if output[1] == oracle[1]:
            right += 1
        else:
            wrong += 1
        error_distance = calc_dist_degree(output[2], output[3], oracle[2], oracle[3])
        error_distance_list.append(error_distance)

    accuracy = round(right / (right + wrong + 1e-6), 3)
    error_distance_list.sort()
    total_num = len(error_distance_list)
    mean_error = round(sum(error_distance_list) / (total_num + 1e-6), 1)
    median_error = round(error_distance_list[int(total_num / 2)], 1)
    result = "{}& {}& {}& {}& {}".format(output_file,
                                         submission_type,
                                         accuracy,
                                         median_error,
                                         mean_error)
    print(result)


def etl_data(data_file, submission_type):
    etl_data_list = []
    key = "hashed_{}_id".format(submission_type.lower())
    with open(data_file) as fr:
        for l in fr:
            data_item = json.loads(l)
            hid = data_item.get(key)
            city = data_item.get("city")
            lat = data_item.get("lat")
            lon = data_item.get("lon")
            tup = (hid, city, lat, lon)
            assert all(tup)
            etl_data_list.append(tup)
    etl_data_list = sorted(etl_data_list, key=lambda k: k[0])
    return etl_data_list


def main():
    parser = argparse.ArgumentParser(description="Evaluation scripts for WNUT 2016 shared task.\
    Output format: submission file, submission type, accuracy, median error distance, mean error distance.")
    parser.add_argument('--output', dest='output', required=True, help='Submission JSON file')
    parser.add_argument('--oracle', dest='oracle', required=True, help='Labelled oracle data JSON file')
    parser.add_argument('--type', dest='type',
                        required=True,
                        choices=["TWEET", "USER"],
                        help='Submission type, either TWEET or USER')
    args = parser.parse_args()
    evaluate_submission(args.output, args.oracle, args.type)


if __name__ == "__main__":
    main()
