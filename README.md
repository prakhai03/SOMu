# SOMu

**SOMu** is an online compiler and visualizer for Self-Organizing Maps (SOMs). It provides a web-based interface to train, visualize, and interact with SOMs, facilitating a deeper understanding of high-dimensional data through unsupervised learning techniques.

## Features

- **Interactive Visualization**: Visualize the training process and results of Self-Organizing Maps in real-time.
- **Web-Based Interface**: Access the tool through a browser without the need for local installations.
- **Modular Architecture**: Separation of frontend and backend components for scalability and maintainability.
- **Cross-Platform Support**: Compatible with major operating systems through platform-specific scripts.

## Project Structure

The repository is organized into the following main directories:

- `backend/`: Contains the server-side code responsible for processing SOM algorithms and handling API requests.
- `frontend/`: Houses the client-side application built with modern web technologies for user interaction and visualization.
- `dist/`: Distribution files for deployment purposes.
- `node_modules/`: Dependencies for the frontend application.
- `SOMPY.egg-info/`: Metadata for the Python package.

Additionally, there are several scripts to facilitate setup and execution:

- `setup.bat` / `setup.sh`: Scripts to set up the project environment on Windows and Unix-based systems, respectively.
- `install-deps.bat`: Installs necessary dependencies.
- `run-backend.bat`: Starts the backend server.
- `start-frontend.bat`: Launches the frontend application.
- `start.bat`: Initiates both frontend and backend components.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/prakhai03/SOMu.git
cd SOMu
