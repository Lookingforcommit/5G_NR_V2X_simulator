from datetime import datetime
from matplotlib import pyplot as plt

import exceptions
from io_processing import UserInput
from calculations import CalcProc
from plotting import GraphPlotter


if __name__ == '__main__':
    start_time = datetime.now()
    input_handler = UserInput()
    try:
        parsed_data = input_handler.start()
        config, data = parsed_data[0], parsed_data[1]
        calc_proc = CalcProc(data, config['envScenario'], config['signalFreq'], config['signalPower'],
                             config['receptionThreshold'])
        plotter = GraphPlotter(calc_proc.dists, config['propLossRegion'], config['packagesRegion'])
        for i in range(config['simCnt']):
            prop_loss = calc_proc.calc_prop_loss()
            signal_reception = calc_proc.calc_signal_reception(prop_loss)
            plotter.generate_distance_data_dependencies(prop_loss, signal_reception)
        plotter.plot_metrics()
        total_time = datetime.now() - start_time
        print(f'Время симуляции: {total_time.total_seconds()} секунд')
        plt.show()
    except exceptions.DataParsingException as exc:
        print(exc)
