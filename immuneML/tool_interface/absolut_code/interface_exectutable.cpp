// New executable version
// This version uses the source code from Absolut

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <cstring>
#include <unistd.h>
#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "standalone_main_functions.h"

using namespace std;
using json = nlohmann::json;

struct InputData {
    string option;
    string antigen;
    string filename;
    int threads;
};

InputData create_inputData() {
    InputData newInputData;
    return newInputData;
}

// Handles the input data received from immuneML by filling in an InputData struct
InputData getJSON(std::string jsonString) {
    // Parse the JSON string
    json j = json::parse(jsonString);

    std::cout << j << std::endl;

    // Get the values
    InputData inputData;
    inputData.option = j["option"].get<std::string>();
    inputData.antigen = j["antigen"].get<std::string>();
    inputData.filename = j["filename"].get<std::string>();
    inputData.threads = j["threads"].get<int>();

    std::cout << "Printing values received from immuneML in Absolut" << std::endl;
    std::cout << "  - Option: " << inputData.option << std::endl;
    std::cout << "  - Antigen: " << inputData.antigen << std::endl;
    std::cout << "  - Filename: " << inputData.filename << std::endl;
    std::cout << "  - Threads: " << inputData.threads << std::endl;

    return inputData;
}

// This is not the most optimal solution, but because option2 does not return the name of the produced filed
// this is a temporary solution
std::string getDatasetPath(std::string antigen) {
    // Get the directory we are currently in
    std::string currDirPath = "";
    char buffer[PATH_MAX];
    if (getcwd(buffer, PATH_MAX) != nullptr) {
        currDirPath = buffer;
    }

    const char* dirPath = currDirPath.c_str();
    std::string filePath = "";
    DIR* dir = opendir(dirPath);

    if (dir) {
        dirent* file;

        // Substrings used to identify the dataset file produced
        const char* filenameSubstring = "FinalBindings_Process_";
        const char* antigenSubstring = antigen.c_str();

        // Search for file based on substrings
        while ((file = readdir(dir)) != nullptr) {
            if ((strstr(file->d_name, filenameSubstring) != nullptr) &&
                (strstr(file->d_name, antigenSubstring) != nullptr)) {
                filePath = std::string(dirPath) + "/" + std::string(file->d_name);
                std::cout << "FOUND THE FILEPATH: " << filePath << std::endl;
            }
        }
        closedir(dir);
    }

    return filePath;
}

int main(int argc, char* argv[]) {
    std::cout << "Running exectuable V2 for Absolut" << std::endl;

    if (argc < 3) {
        std::cout << "JSON string with parameters and a port number required" << std::endl;
        return 1;
    }

    // Handle input data
    InputData inputData = getJSON(argv[1]);
    std::string port_number = argv[2];

    std::string port_string = "tcp://*:" + port_number;

    // ZeroMQ
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    std::cout << "Connecting to port: " << port_string << std::endl;
    socket.bind(port_string);

    // Call for option2 to generate dataset
    option2(inputData.antigen, inputData.filename, inputData.threads);

    // Send the dataset path as response to immuneML
    std::string dataset_path = getDatasetPath(inputData.antigen);
    zmq::message_t reply(dataset_path.size());
    memcpy(reply.data(), dataset_path.data(), dataset_path.size());
    socket.send(reply);


    return 0;
}
