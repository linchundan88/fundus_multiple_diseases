
1.License
This software is under the GNU GENERAL PUBLIC LICENSE V2.

2.Project background
This project is part of the automated Retinopathy of Prematurity screening platform.
The platform contains three parts: training and validation, RPC service and web application.
This project contains the first two parts of the platform.
And the third part(web application) can be found at the project "ROP_Web".

3.Code structure
The LIBS sub-directory includes shared libraries.
The RPC sub-directory provide RPC services, which can be called by the web application


This project contains some deprecated and redundant codes.
For the classification task, the codes at project "DR" are more concise and clear, even though the code style and neural network models are the same. The project "DR" use tensorflow2.2 instead of tensorflow1.x.
