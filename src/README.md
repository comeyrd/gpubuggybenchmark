# gpubuggybenchmark 

## How to add a new Kernel

You will need to implement two interfaces and register theses to an implementation manager.

So you'll need to implement three structs, with theses requirements : 

Data and Result must be only instantiated with a const settings as a parameter. 
The instantiation of Data and result must be explicit.

Data must have a generate_random() method that will fill the object with random values.

The Settings needs at least : int repetitions and int warmup values.

Then you'll need to implement this interface : 
class NewKernel : public IKernel<NewData, NewSettings, NewResult>

You'll only need to override the run_cpu() method. you'll need to put the result inside the cpu_result value.

Then you'll need to register the Implementation you just did in a cpp file : 

REGISTER_CLASS(I_IKernel, NewKernel);

Then to add gpu versions of this kernel, you can create this subtype : 

using INewVersion = IVersion<NewData,NewSettings,NewResult>;

Then, each version will need to implement the run with the types you just implemented.

Then register each version with : 
REGISTER_CLASS(INewVersion,ReferenceNewVersion);