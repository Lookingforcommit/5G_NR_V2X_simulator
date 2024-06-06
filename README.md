## User Manual
This project has a Russian [documentation](https://www.overleaf.com/read/vdztcprygdmj#011a6f) on overleaf
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

After performing these steps, you should have a FILENAME.csv file of the SUMO FCD format in your working directory

![5G_NR_V2X_simulator](/images/FCD_processed.png)

### 3\. Starting the program

1\. Create a simulation configuration file of the following format:

Key1=Value

Key2=Value

**dataFilepath** - full path to the simulation file FILENAME.csv

**csvSep** - csv file separator, default option is *semicolon*

**envScenario** - simulation scenario, available options - *highway*. The default scenario is *highway*

**signalFreq** - frequency of the signal transmitted by the machines, positive integer

**signalPower** - power of the signal transmitted by the machines, positive integer

**propLossRegion** - the averaging regions for propagation loss chart, positive integer. The default value is *500*

**packagesRegion** - the averaging regions for all the packages-based metrics, positive integer. The default value 
is *500*

**simCnt** - number of simulation runs, positive integer. The default value is *1*

**receptionThreshold** - minimal signal power required for the machine receiver to process a message, float

The program can correctly process simulation files of extensions *.csv*, *.xls* and *.xlsx*, an attempt to transfer 
a file of other extension will cause an error.

2\. Run the program through the command line

**Important:** the program perceives the path to the simulation data file relative to the script position in the file
system, so if your configuration file is not in the same folder as the program, the absolute path to the simulation data
file must be passed as the **dataFilepath** parameter.

### 4\. Analysing simulation results
The program returns a set of signal reception metrics, which show how effective the cars were at communicating 
with each other.

![5G_NR_V2X_simulator](/images/metrics/PL.eps)

![5G_NR_V2X_simulator](/images/metrics/PLR.eps)

![5G_NR_V2X_simulator](/images/metrics/PRR.eps)
