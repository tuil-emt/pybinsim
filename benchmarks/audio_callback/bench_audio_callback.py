from multiprocessing.connection import Connection
import numpy as np
from pythonosc import udp_client
from time import perf_counter_ns, sleep
import multiprocessing
from multiprocessing import Event, Pipe, Value, Process

def osc_message_load(stop_flag: Event, messages_per_second: Value, start_message_timing: Event, tx: Connection):
    """ Switches azimuth between -180 and -178, until message timing is started, 
    then azimuth rises from -176 to 178 and times of sending the messages are saved and sent over the pipe once done. 
    The message load then switches back to flipping between -180 and -178 azimuth.
    """
    print("STARTING OSC LOAD CLIENT")
    client = udp_client.SimpleUDPClient("127.0.0.1", 10000)
    azimuths = range(-180, 179, 2)
    filter_keys = [[0, azimuth, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, ] for azimuth in azimuths]
    times_sent: list[tuple[int,int]] = [(0,0)] * 178 # (azimuth, perf_counter_ns)
    i = 0
    j = 0
    while not stop_flag.is_set():
        if messages_per_second.value > 0:
            sleep(1/messages_per_second.value)
        else:
            sleep(0.05)
            continue
        client.send_message("/pyBinSim_ds_Filter", filter_keys[i])
        if start_message_timing.is_set() and azimuths[i] > -178:
            times_sent[j] = (azimuths[i], perf_counter_ns())
            j += 1
        i += 1
        if start_message_timing.is_set():
            if i == len(filter_keys):
                i = 0
                j = 0
                start_message_timing.clear()
                tx.send(times_sent)
        else:
            i = i % 2
    print("STOPPING OSC LOAD CLIENT")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    N = 2_000
    message_loads = [0, 1000, 10000]
    times = list()
    times_azimuth_sent = list()
    times_azimuth_received = list()

    import pybinsim
    binsim = pybinsim.BinSim(
        'benchmarks/audio_callback/pyBinSimSettings_audio_callback.txt')


    class MockStream:
        def __init__(self):
            self.cpu_load = 0

        def close(self):
            pass


    binsim.stream = MockStream()
    callback = pybinsim.application.audio_callback(binsim)
    output_buffer = np.zeros((512, 2))




    stop_flag = Event()
    start_message_timing = Event()
    messages_per_second = Value("I", 0)
    rx, tx = Pipe(duplex = False)

    load_process = Process(target=osc_message_load,
                        args=(stop_flag, messages_per_second, start_message_timing, tx))
    load_process.start()


    for msgs in message_loads:
        print(f"Messages per second: {msgs}")
        messages_per_second.value = msgs
        condition_times = np.zeros(N)  # in ms
        sleep(0.1) # wait for message rate to settle
        for i in range(N):
            start = perf_counter_ns()
            callback(output_buffer, None, None, None)
            stop = perf_counter_ns()
            elapsed = stop - start  # in ns
            condition_times[i] = elapsed/1e6  # in ms
            if i == 0 and msgs > 0:
                start_message_timing.set()
        times.append(condition_times)
        if msgs > 0:
            sleep(1) # wait for all messages to get handled
            times_azimuth_sent.append(rx.recv())
            times_azimuth_received.append(binsim.oscReceiver.get_times_azimuth_received_and_reset())


    stop_flag.set()
    load_process.join()

    binsim.__exit__(None, None, None)

    print("")
    print("Audio Callback Duration in ms")
    print("=============================")
    print(f"{N} samples for each message load")

    headings = ["msg/s", "mean", "std", "min", "50 %",
                "99 %", "max4", "max3", "max2", "max", "first"]
    heading_format = "{:>8} " * (len(headings))
    print(heading_format.format(*headings))

    row_format = "{:>8.2f} " * (len(headings))


    def print_times(times, msg_per_sec):
        sorted = np.sort(times)
        row = [
            msg_per_sec,
            np.mean(times),
            np.std(times),
            *np.percentile(times, (0, 50, 99)),
            *sorted[-4:],
            times[0],
        ]
        print(row_format.format(*row))


    for i in range(len(message_loads)):
        print_times(times[i], message_loads[i])

    print("")
    print("OSC Message Sent to Handled Duration in ms")
    print("==========================================")

    message_times = list()
    out_of_orders = list()
    for i in range(len(times_azimuth_sent)):
        message_times_for_load = list()
        out_of_order = 0
        for j in range(len(times_azimuth_sent[i])):
            azimuth_sent, sent = times_azimuth_sent[i][j]
            try:
                matching_ind = next(k for k in range(len(times_azimuth_received[i])) if times_azimuth_received[i][k][0] == azimuth_sent)
                if matching_ind != j:
                    out_of_order += 1
                received = times_azimuth_received[i][matching_ind][1]
                message_times_for_load.append((received - sent)/1e6)
            except StopIteration:
                pass # message not received, ignore
        message_times.append(message_times_for_load)
        out_of_orders.append(out_of_order)

    for i in range(len(times_azimuth_sent)):
        print(f"{message_loads[i+1]} msg/s: {len(times_azimuth_sent[i])} samples sent, {len(times_azimuth_received[i])} received, {out_of_orders[i]} out of order")
    print(heading_format.format(*headings))

    for i in range(len(times_azimuth_sent)):
        if len(message_times[i]) >= 4:
            print_times(message_times[i], message_loads[i+1])
        else:
            print(f"{message_loads[i+1]:>8.2f}     not enough messages received for statistics")
