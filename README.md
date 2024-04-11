## User Manual
### 1\. Setting up and starting SUMO
1\. Create a road network file, save the config file and road flow settings.

![5G_NR_V2X_simulator](/images/SUMO_highway.png)

2\. In the simulation configuration file (extension .sumocfg) add an instruction to save the simulation results in the FCD (flying cars data) format.

	<output>
		<fcd-output value="FILENAME.xml" />;
	</output>

This instruction must be inserted between the &lt;configuration&gt; and &lt;/configuration&gt; tags

### 2\. Converting SUMO data

1\. Open terminal and, using the cd command, navigate to the directory with the simulation result.

2\. Run the command *python SUMOHOME/tools/xml/xml2csv.py FILENAME.xml*.

After performing these steps, you should have a FILENAME.csv file of this format in your working directory

![5G_NR_V2X_simulator](/images/FCD_processed.png)

### 3\. Starting the program

1\. Create a simulation configuration file in .txt format

The configuration file should have the following format:

Key1=Value

Key2=Value

Valid simulation parameters:

**dataFilepath** - full path to the simulation file FILENAME.csv

**csvSep** - csv file delimiter, default option is *semicolon*

**signalFreq** - frequency of the signal transmitted by the machines, integer

**signalPower** - power of the signal transmitted by the machines, integer

**receptionThreshold** - minimal signal power requied for the machine receiver to process a message, float

**envScenario** - simulation scenario, available options - *highway*, *urban*. The default scenario is *highway*.

**propLossRegion** - the averaging regions for propagation loss chart, integer

**packagesRegion** - the averaging regions for all the packages-based metrics, integer

The program can correctly process simulation files of extensions *.csv*, *.xls* and *.xlsx*, an attempt to transfer a file of other extensions will cause an error.

The simulation file must be of the SUMO FCD (flying cars data) format, otherwise the program will not be able to run the simulation.

2\. Run the program through the command line

**Important:** the program perceives the path to the simulation data file relative to its position in the file system, so if your configuration file is not in the same folder as the program, the absolute path to the simulation data file must be passed as the **dataFilepath** parameter.

### 4\. Analysing simulation results
The program returns a set of signal reception metrics, which show how effective the cars were at communicating with each other.

![5G_NR_V2X_simulator](/images/metrics.png)
