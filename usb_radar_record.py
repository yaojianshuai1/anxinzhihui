#!/usr/bin/env python3
import argparse
import csv
import json
import struct
import time
from datetime import datetime

import serial


def calculate_checksum(data: bytes) -> bytes:
    return struct.pack('>H', sum(data) & 0xFFFF)


def build_packet(cmd: int, data: bytes = b'') -> bytes:
    header = b'\x55\xaa'
    return header + struct.pack('B', cmd) + struct.pack('B', len(data)) + data + calculate_checksum(header + struct.pack('B', cmd) + struct.pack('B', len(data)) + data)


def parse_packet(packet: bytes):
    if len(packet) < 6 or packet[0:2] != b'\x55\xaa':
        return None
    cmd = packet[2]
    ln = packet[3]
    if len(packet) < 6 + ln:
        return None
    data = packet[4:4 + ln]
    recv_ck = packet[4 + ln: 6 + ln]
    calc_ck = calculate_checksum(packet[:4 + ln])
    ok = recv_ck == calc_ck
    parsed = {
        'cmd': cmd,
        'len': ln,
        'data_hex': data.hex(),
        'ck_ok': ok,
    }
    if ln >= 32:
        try:
            uid = data[0:6].hex()
            heart = data[6]
            resp = data[7]
            presence = data[8]
            activity = data[9]
            distance_cm = struct.unpack('>H', data[10:12])[0]
            signal_strength = struct.unpack('>f', data[13:17])[0]
            in_bed_minutes = struct.unpack('>H', data[17:19])[0]
            out_bed_minutes = struct.unpack('>H', data[19:21])[0]
            hour, minute, second = data[24], data[25], data[26]
            parsed.update({
                'uid': uid,
                'heart_raw': heart,
                'resp_raw': resp,
                'presence': presence,
                'activity': activity,
                'distance_cm': distance_cm,
                'signal_strength_db': signal_strength,
                'in_bed_minutes': in_bed_minutes,
                'out_bed_minutes': out_bed_minutes,
                'device_time': f'{hour:02d}:{minute:02d}:{second:02d}',
            })
            # candidate interpretations
            parsed['heart_candidates'] = {
                'raw': heart,
                '/10': heart / 10.0,
                '/100': heart / 100.0,
                'signed': struct.unpack('b', struct.pack('B', heart))[0]
            }
            parsed['resp_candidates'] = {
                'raw': resp,
                '/10': resp / 10.0,
            }
            parsed['distance_candidates'] = {
                'cm': distance_cm,
                'm/100': distance_cm / 100.0,
                'm/1000': distance_cm / 1000.0,
            }
        except Exception:
            pass

    return parsed


def record(port, baud, duration, out_path):
    ser = serial.Serial(port, baud, timeout=0.5)
    # send initial probes
    ser.write(build_packet(0x00))
    ser.flush()
    time.sleep(0.05)
    ser.write(build_packet(0x01))
    ser.flush()

    end = time.time() + duration
    buf = b''
    rows = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ts', 'raw_hex', 'cmd', 'len', 'ck_ok', 'data_hex', 'uid', 'heart_raw', 'resp_raw', 'distance_cm', 'signal_strength_db', 'in_bed_minutes', 'out_bed_minutes', 'device_time', 'heart_candidates', 'resp_candidates', 'distance_candidates', 'presence', 'activity'])
        writer.writeheader()
        while time.time() < end:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
            # extract packets
            while True:
                idx = buf.find(b'\x55\xaa')
                if idx < 0:
                    buf = b''
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < 6:
                    break
                ln = buf[3]
                pkt_len = 6 + ln
                if len(buf) < pkt_len:
                    break
                pkt = buf[:pkt_len]
                parsed = parse_packet(pkt)
                ts = datetime.utcnow().isoformat() + 'Z'
                raw_hex = pkt.hex()
                row = {
                    'ts': ts,
                    'raw_hex': raw_hex,
                    'cmd': f"0x{pkt[2]:02x}",
                    'len': parsed['len'] if parsed else None,
                    'ck_ok': parsed['ck_ok'] if parsed else False,
                    'data_hex': parsed['data_hex'] if parsed else '',
                    'uid': parsed.get('uid') if parsed else '',
                    'heart_raw': parsed.get('heart_raw') if parsed else None,
                    'resp_raw': parsed.get('resp_raw') if parsed else None,
                    'distance_cm': parsed.get('distance_cm') if parsed else None,
                    'signal_strength_db': parsed.get('signal_strength_db') if parsed else None,
                    'in_bed_minutes': parsed.get('in_bed_minutes') if parsed else None,
                    'out_bed_minutes': parsed.get('out_bed_minutes') if parsed else None,
                    'device_time': parsed.get('device_time') if parsed else None,
                    'heart_candidates': json.dumps(parsed.get('heart_candidates')) if parsed and parsed.get('heart_candidates') else '',
                    'resp_candidates': json.dumps(parsed.get('resp_candidates')) if parsed and parsed.get('resp_candidates') else '',
                    'distance_candidates': json.dumps(parsed.get('distance_candidates')) if parsed and parsed.get('distance_candidates') else '',
                    'presence': parsed.get('presence') if parsed else None,
                    'activity': parsed.get('activity') if parsed else None,
                }
                writer.writerow(row)
                rows += 1
                f.flush()
                buf = buf[pkt_len:]

    ser.close()
    print(f"Saved {rows} packets to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--duration', type=int, default=60)
    parser.add_argument('--out', default='/userdata/camera_photos/radar_log.csv')
    args = parser.parse_args()
    record(args.port, args.baud, args.duration, args.out)


if __name__ == '__main__':
    main()
