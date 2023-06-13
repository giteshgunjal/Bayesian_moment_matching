momentmatching : momentmatching.cpp momentmatching.h
	g++ -o momentmatching momentmatching.cpp  -O3 -I/opt/ros/noetic/include -L/opt/ros/noetic/lib -lroscpp -lrostime -lrosconsole -lroscpp_serialization -ltf2_ros -DEIGEN_STACK_ALLOCATION_LIMIT=0
