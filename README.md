# DS-SLAM-to-Libtorch
一个基于Libtorch的DS-SLAM复现

我复现了DS-SLAM，并且将语义分割框架由caffe替换为了Libtorch，此外，我还删除了ROS与稠密建图的内容，使得我们的关注点放在SLAM本身

你需要的环境支持：Ubuntu、CUDA、Libtorch、ORB-SLAM2的相关环境

运行：在运行前打开Examples/RGB-D/rgbd_tum.cc，确保主函数中的各路径配置正确，其中模型文件见[模型文件](https://pan.baidu.com/s/19FI9tc-0m2t8CmpTB98jFQ?pwd=utcl 
)提取码：utcl，此外，还应重新编译Thirdparty内的g2o与DBoW2，随后在主目录下新建build文件夹并进入编译项目，总之，与ORB-SLAM2操作大体相同，只不过多了加入语义分割模型路径这一步骤。


