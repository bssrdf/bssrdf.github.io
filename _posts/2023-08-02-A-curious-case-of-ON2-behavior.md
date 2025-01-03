# A curious case of O(N^2) behavior which should be O(N)

## Motivation

Recently I got interested in Blender 3D, partly inspired by [infinigen project](https://github.com/princeton-vl/infinigen).

One day I encountered [Tellusim](https://tellusim.com/). Impressed by the quality of its rendering, I was browsing its blogs and see [this](https://tellusim.com/import/). Wow, Tellusim really blowed others out of the water;others including Unreal, Unity, Omniverse and **Blender**. Wait, Blender is really that slow importing a USD scene? 

Since Blender is open-source, why not try to figure out what's going on? Here we go.

## First, let's profile it

I cloned the latest version of Blender 4.0.0.Alpha and installed all dependencies. Building is a straightforward thing. I am working on Windows 10 and miss ```perf``` on Linux. But VS2022 community edition does include a profiling tool: ```Microsoft (R) VS Standard Collector```. Go to the directory where blender.exe is located and issue:
```
VSDiagnostics start 1 /launch:blender.exe /loadConfig:"C:\Program Files\Microsoft Visual Studio\2022\Community\Team Tools\DiagnosticsHub\Collector\AgentConfigs\CpuUsageBase.json"
```

Now in ```blender```, import ```limits_32.usdc``` unzipped from this [test file](https://tellusim.com/download/import/limits.zip). It's going to take a while and on my machine that is about **30** seconds. Once importing is done, run
```
VSDiagnostics.exe stop 1 /output:"usd_import1"
```

File ```usd_import1.diagsession``` will have profiling information which you can examine in VS by opening it. 

![alt](https://gist.github.com/bssrdf/397900607028bffd0f8d223a7acdce7e/raw/61bcd1ea0fe9f94f11476c1fc49ca62b8cfb51b9/profiling1.png) 
This is what it looks like on my system. If you want, a flame graph is also available. 
![alt](https://gist.github.com/bssrdf/397900607028bffd0f8d223a7acdce7e/raw/61bcd1ea0fe9f94f11476c1fc49ca62b8cfb51b9/profiling2.png)

## Analysis

Now, it should be clear that most of the time (28s-53s on the time line) of importing that particular USD file is spent in ```id_sort_by_name```. But why? What exactly does ```id_sort_by_name``` do? It took me a while to figure out with help from the comments in the code. Blender creates many objects (its type is ```ID```) when importing scenes or users creating primitives. Internally, Blender maintains a doubly linked list of these ID's. The list is sorted by ID's **unique** name alphebatically. Everytime a new object (ID) is created, ```id_sort_by_name``` is called to inserted the newly created ID into the list. Because the list is sorted and Blender assumes all objects created later have higher extension number (i.e., alphebatically bigger), the search is done backwards starting from the end. Fine, this sounds reasonable as each insertion most likely happens at the list end which is O(1) and the whole process is O(N) with N being number of new IDs. But how does profiling tell a different story?     

All the usd files having performance issue have something in common: they have one entity shared by many objects. Blender's USD import utility creates a new object for each reference. Because they all have the same name, another process came in to assign a number to objects which have names already registered to make them unique. The problem is the assignment is based on a counter starting from 1. It ends up like this: an ID with name ```basename.12313``` is alphabetically smaller than ```basename.9999``` although the latter comes later than the former. So the linked list search needs to take a lot more steps to find a place to insert ```basename.12313```. Effectivelly, an O(N) process becomes O(N^2). We all know how bad O(N^2) is, right?          

## Solution

~~Up to now, the solution should also be clear: change the naming methods. Instead of changing the general name mangling function Blender uses, I made some modification directly to the USD import module: I count the total number of objects in a USD file and get the width of total digits, then for each unique basename, assign a new name with the format ```basename_00XXX``` where XXX is their index and use 0 padding so they all have the same number of digits. This way ```ID_00999``` will be near the head of the list, not the tail.~~

The above solution turns out to be not ideal. It couldn't handle importing the same USD file over and over or multiple ones all with the same basename(back to O(N^2) very quickly). Given that the process of assigning extension number uses a counter for each basename, I came out another solution. In ```id_sort_by_name```, instead of comparing the whole name (basename + number), I split it into two parts: first compare basename; and if basename is the same, compare number. This fix gave similar performance as before (see next), while allowing importing arbitrary number of files.

## Results

After the above ad-hoc fix, loading ```limits_32.usdc``` took **8** seconds and a even bigger file ```limits_48.usdc``` took **27** seconds while before fix it spent a whole **4 min 40 sec**. Not bad. 
![alt](https://gist.github.com/bssrdf/397900607028bffd0f8d223a7acdce7e/raw/61bcd1ea0fe9f94f11476c1fc49ca62b8cfb51b9/profiling3.png)
Now ```id_sort_by_name``` is not the bottleneck anymore. Other processes stand out, e.g. ```USDMeshReader::read_object_data```. For bigger files, building the DAG becomes more time consuming. There may be room to further cut down some time (in another post maybe) such that the whole Blender USD import pipeline matches ```Omniverse 2023.1```. However, I doubt Blender is ever going to reach the level close to ```Tellusim```.


Thanks for reading.
















