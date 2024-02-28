from functions import *


if __name__ == '__main__':
    signal_levels = 2
    downsampling_factor = 1
    initial_sps = 256

    eye_diagram_window = 10  # %
    roll_samples = 0
    cut_last_symbols = 0
    roll_search_window = 30  # %
    ideal_roll_search_speed_up = 4

    data_folder = 'Signal Data'
    file_name = '...'

    check_directory_exists(file_name, data_folder)

    R_frame = check_file_exists(data_folder, file_name)
    signal = R_frame.to_numpy().flatten()

    signal = downsampler(signal, downsampling_factor)
    updated_sps = initial_sps // downsampling_factor

    signal = cut_signal_samples(signal, cut_last_symbols, updated_sps)
    signal = np.roll(signal, roll_samples)
    plot_signal(signal)

    ideal_signal_roll = maximize_q_factor(signal, signal_levels, updated_sps, roll_search_window, ideal_roll_search_speed_up)
    signal = np.roll(signal, ideal_signal_roll)

    draw_eye_diagram(signal, updated_sps, eye_diagram_window, file_name)
    calculate_q_factor(signal, signal_levels, updated_sps, eye_diagram_window, file_name)
