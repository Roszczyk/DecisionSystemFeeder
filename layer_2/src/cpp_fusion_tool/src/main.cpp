#include <iostream>
#include <string>

struct State {
    std::string name;
    float mass;
};

int main(int argc, char* argv[]) {
    if (argc < 2)
        std::cout << "Please provide arguments" << std::endl;
    /*
    Idea for arguments:
    argv[1] == name of the method, e.g. DST
    argv[2] == available states, e.g. "A,B,C"
    argv[3] == mass for each of the 2^X sets - "[A],[B],[C],[AB],[AC],[BC],[ABC]",
                    e.g. "0.1,0.2,0.5,0.0,0.1,0.1,0"
    */
    State state;
}