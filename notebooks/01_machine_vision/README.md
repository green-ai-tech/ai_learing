1. python  3.13.9

2. pip

3. python的安装路径：
	管理员安装：C:\Program Files\Python313
	用户安装：C:\Users\ThinkPad\AppData\Roaming

4. torch,hf的数据集 + 模型下载的位置：
	C:\Users\ThinkPad\.cache
	
	自己下载：
		|- https://hf-mirror.com/   ： https://huggingface.co/
		|- https://modelscope.cn/
		|- https://modelscope.cn/models


5. 开发工具：
	vscode：编辑器（批处理编程：非交互式编程）（编码）： 面向对象。
	jupyter：编辑器（交互式编程）：记事本notebook
	cmd：（不使用power shell:模仿linux）
		|- 当前用户模式（权限低）： 安装模块在C:\Users\ThinkPad\AppData\Roaming\python313
		|- 管理员模式（权限高）：C:\Program Files\Python313

pip install jupyter 
	pip show jupyter:查看安装模块
	更多的pip功能：pip --help

6. 进入jupyter一定切换到工作目录
	cd 目录
	盘符:
	使用tab进行命令补全
	使用上下方向键，可以切换历史命令。

7. 补充：
	window是怎么执行程序的？
		| - 根据命令，确定执行程序文件名jupyter -> jupyter.exe  jupyter.bat
        | - 在当前找程序
        | -  windows/system32
        | - 在环境变量（用户，系统），Path中指定的路径找。
        | - 如果找到，加载执行。
        | - 如果找不到：'jupyter' 不是内部或外部命令，也不是可运行的程序或批处理文件。
        		|- 外部命令：存在执行程序的命令
        		|- 内部命令：不存在文件，在系统启动的时候，内存中存在。
        问题解决：
        	| - 指知道安装路径，设置path环境变量（用户，系统[管理员]）
        		设置->系统->系统信息->高级系统设置->环境变量->用户环境/系统环境变量：path。

        	注意：
        		终端重启


8. 语言分两个体系：algo(SQL,COBOLPASCAL,Delpph,Basic,VB),B(C,C++,Java,OC) 

   