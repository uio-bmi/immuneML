/*
This file contains the option2 function belonging to the main functionality of Absolut
This file is necessary to be able to use the function inside the interface_executable
*/

#include "standalone_main_functions.h"

#include "../Ymir/ymir.h"
#include "../Tools/md5.h"
#include "../Absolut/motifFeatures.h"
#include "../Absolut/quality.h"
#include "../Absolut/selfEvo.h"
#include "../Tools/zaprandom.h"
#include "../Tools/dirent.h"
#include <regex>
#include "../Absolut/antigenLib.h"
#include "../Absolut/importrepertoire.h"
#include "../Absolut/poolstructs.h"
#include "../Absolut/epitope.h"
#include "../Absolut/html.h"
#include "../Absolut/fileformats.h"
#include "../Absolut/topology.h"
#include "../Absolut/dlab.h"
#include <iostream>
#include <fstream>

#include <iostream>
using namespace std;




int testing_return() {
    std::cout << "Hey there broski" << std::endl;
    return 0;
}



// -------------------------------- CODE FROM ABSOLUT --------------------------------
// for absolut this is 10 and 11
#define DefaultReceptorSizeBonds 10
#define DefaultContactPoints 11

static pthread_mutex_t lockSaveCommonDataset =  PTHREAD_MUTEX_INITIALIZER;

// ================= Option 2: Multithread repertoire binding calculation =====================.

// In order to use threads, the computation function needs to receive only one argument as void*
// therefore, we pool all the arguments into one structure.
// http://www.cse.cuhk.edu.hk/~ericlo/teaching/os/lab/9-PThread/Pass.html
struct argsThread {
    argsThread(){}
    affinityOneLigand* T3;
    vector< std::pair<string, string> >* listToProcess;
    string resultFile;
    string identificationThread;
    string antigenName;
    int receptorSize;
    dataset<binding>* savingLocation;
};



// Main function for the repertoire option, that will be run on each single thread separately.
void *oneThreadJob(void *input){
    // 1 - Reconstituting the arguments from the pointer to the class argsThreads
    string resultFile = ((struct argsThread*)input)->resultFile;
    int receptorSize = ((struct argsThread*)input)->receptorSize;
    affinityOneLigand* T3 =  ((struct argsThread*)input)->T3;
    vector< std::pair<string, string> >* listToProcess = ((struct argsThread*)input)->listToProcess;
    string antigenName =  ((struct argsThread*)input)->antigenName;
    string identificationThread = ((struct argsThread*)input)->identificationThread;
    dataset<binding>* savingLocation = ((struct argsThread*)input)->savingLocation;

    // 2- Preparing output by two ways:
    // a/ writing in a file. It was problems by keeping the ofstream opened, so I will close it each time and do append
    // first, erases the file if existed
    {
        ofstream fout(resultFile.c_str()); //, ios::app);
        if(!fout) {
            cerr << "ERR: Couldn't open output file: " << resultFile << endl;
            return nullptr;
            //pthread_exit(nullptr); => This is bad because if only one thread, it would destroy the main thread...
        }
        fout << antigenName << "\n";
        fout.close();
    }
    string blockToWrite;    // stores output by blocs to write 100 lines by 100 lines into the result files
                            // Note: it ended up impossible to clear a stringstream, that's why I store as a string,
                            // but i build by blocks using stringstream (see blocksOut)

    // Treats each CDR3 sequence (from the list to process by this thread)
    size_t Nseq = listToProcess->size();
    cout << "Thread " << identificationThread << " got " << Nseq << " sequences to proceed " << endl;

    for(size_t i = 0; i < Nseq; ++i){
        std::pair<string, string> next = listToProcess->at(i);
        string CDR3seq = next.second;
        string ID = next.first;

        // cout << "Process " << next.first << "=> " << next.second << endl;
        if(CDR3seq.size() > 0){
            stringstream blocksOut;
            //cout << "Thread " << identificationThread << " treats ID " << ID << endl;
            vector<string> cutSlides = slides(CDR3seq, receptorSize+1);
            size_t nSl = cutSlides.size();

            // 1st step: calculate the affinities of each sliding window and finds the best(or equally best) slides
            // Note: it's not very optimal to duplicate affinities, Luckily, the affinity() function remembers previously calculated ones
            double bestEnergy = +1000000;
            for(size_t si = 0; si < nSl; ++si){
                string subSeq = cutSlides[si];
                std::pair<double, double> affs = T3->affinity(subSeq, false);
                bestEnergy = min(affs.first, bestEnergy);
            }

            // 2: Now, knowing who is the best, re-requests affinity of each sliding windows and says 'best' for the equally best ones
            // Since affinities are stored in memory, it will not trigger recalculation
            for(size_t si = 0; si < nSl; ++si){

                string subSeq = cutSlides[si];
                vector<string> optimalStructures;

                // Gets affinity and all optimal bindings (sometimes more than one)
                std::pair<double, double> affs = T3->affinity(subSeq, false, &optimalStructures);

                // Now we know what is the best binding energy for this CDR3
                bool isBest = (fabs(affs.first - bestEnergy) < 1e-6);

                #define commonSaving true
                if(commonSaving) pthread_mutex_lock(&lockSaveCommonDataset);

                size_t nS = optimalStructures.size();
                // For each optimal binding structure
                for(size_t j = 0; j < nS; ++j){

                    string IDthisSlide = combinedID(ID, static_cast<int>(si), static_cast<int>(j));

                    // only says what is happening for the top 10 sequences.
                    if(i < 10) cout << "Thread " << identificationThread << " " << IDthisSlide << "\t" << CDR3seq << "\t" << ((isBest) ? "true" : "false") << "\t" << subSeq << "\t" << affs.first << "\t" << optimalStructures[j] << "\n";

                    blocksOut << IDthisSlide << "\t" << CDR3seq << "\t" << ((isBest) ? "true" : "false") << "\t" << subSeq << "\t" << affs.first << "\t" << optimalStructures[j] << "\n";

                    if(commonSaving) if(savingLocation){
                        binding* b = new binding(subSeq, affs.first, optimalStructures[j]);
                        savingLocation->addLine(IDthisSlide, CDR3seq, b, isBest);
                    }
                }
                if(nS == 0) {
                     blocksOut << combinedID(ID, static_cast<int>(si), 0) << "\t" << CDR3seq << "\t" << ((isBest) ? "true" : "false") << "\t" << subSeq << "\t" << affs.first << "ERR:No_optimal_structure_???" << "\n";
                }
                if(commonSaving) pthread_mutex_unlock(&lockSaveCommonDataset);

            } // end for each sliding window

            blockToWrite += blocksOut.str();
            if(i == 10) cout << "Thread " << identificationThread << "   ... now continues in silent mode ... " << endl;


            // Writing in output file every 100 CDR3 sequences
            if((i % 100) == 0){
                cout << "Thread " << identificationThread << "-- write " << i << "/" << Nseq << endl;
                ofstream fout(resultFile.c_str(), ios::app);
                if(fout){
                    fout << blockToWrite;
                    fout.close();
                    blockToWrite.clear();
                } else {
                    // decided to put errors in cout in threads, because cerr cuts the couts into pieces
                    cout << "Thread " << identificationThread << ", had problems to write into file, keeping into memory till next try" << endl;
                    cout << "Thread " << identificationThread << ", file was:" << resultFile << endl;
                }
            }
        }
    }

    cout << "Thread " << identificationThread << "-- write FINAL " << Nseq << endl;

    {
        ofstream fout(resultFile.c_str(), ios::app);
        if(fout){
            fout << blockToWrite;
            fout.close();
        } else {
            // decided to put errors in cout in threads, because cerr cuts the couts into pieces
            cout << "Thread " << identificationThread << ", had problems to write into file, WILL NOW OUTPUT RESULT " << endl;
            cout << "Thread " << identificationThread << ", file was:" << resultFile << endl;
            cout << blockToWrite;
        }
    }

    //pthread_exit(nullptr);

    // normally this is done by pthreads, but compiler complains reaching end without return
    //cerr << "Will exit thread" << endl;
    return nullptr;
}


// the IDs start at 0
#define MAX_ALLOWED_THREADS 50
//void option2(string ID_antigen, string repertoireFile, int nThreads = 1, string prefix = string(""), int startingLine = 0, int endingLine = 1000000000){
void option2(string ID_antigen, string repertoireFile, int nThreads, string prefix, int startingLine, int endingLine){

    // Default options for our foldings:
    int receptorSize = DefaultReceptorSizeBonds; //10; // defined in number of bounds, add +1 to get the number of AAs
    int minInteract = DefaultContactPoints; //11;

    // nJobs will be the ID of this process (if MPI is used, each job will get a different rank), if not, there is only one job with rank 0
    int nJobs = 1;
    int rankProcess = 0;
    #ifdef USE_MPI
        // If MPI is used (amd compiled with), it will just start independent Jobs, with a certain ID (to split sequences to treat)
        //MPI_Init(nullptr, nullptr);
        MPI_Comm_size(MPI_COMM_WORLD, &nJobs);
        MPI_Comm_rank(MPI_COMM_WORLD, &rankProcess);
        cout << "MPI started!" << endl;
    #endif

    // This mutex control the access to shared memory inside affinityOneLigand::affinity()
    if (pthread_mutex_init(&lockAccessPrecompAffinities, nullptr) != 0){
        cerr << "\nERR: option repertoire, mutex lockAccessPrecompAffinities init failed, problem with pthreads?" << endl;
    }
    // This mutex controls writing the results on the common memory
    if (pthread_mutex_init(&lockSaveCommonDataset, nullptr) != 0){
        cerr << "\nERR: option repertoire, mutex lockSaveCommonDataset init failed, problem with pthreads?" << endl;
    }

    // each process gets an ID that will be used for every communication, because they will output at any time/order
    stringstream IDjob; IDjob << "Job" << rankProcess+1 << "/" << nJobs;
    string myID = IDjob.str();


    // => Checking inputs one by one

    // 0 - Basics
    if(nThreads < 0) {cerr << "ERR: option repertoire, wrong number of threads, will take 1" << nThreads << endl; nThreads = 1;}
    if(nThreads > MAX_ALLOWED_THREADS) {cerr << "ERR: option repertoire, Does not allow more than " << MAX_ALLOWED_THREADS << " threads (requested: " << nThreads << "), will take 50" << endl; nThreads = MAX_ALLOWED_THREADS;}
    if(startingLine > endingLine){cerr << "ERR: option repertoire, the ending line " << endingLine << " is before the starting line " << startingLine << " -> ending" << endl; return;}

    // 1 - Reading the list of sequences to process:
    cout << myID << "   ... loading repertoire " << repertoireFile << "\n";
    //<< "\n       expected to have 2 or 3 columns: ID , CDR3 sequence , [optional tag]" << endl;
    repertoire rep = repertoire(repertoireFile);
    cout << myID << "   ... Found " << rep.nLines() << " lines/sequences " << endl;
    if(rep.nLines() < 1) return;

    // 2a - Loading the antigen from the library. Either call with a number, or with a substring of its name.
    string AntigenName = (ID_antigen.size() < 4) ? IDshortcut(std::stoi(ID_antigen)) : IDshortcut(ID_antigen);
    cout << myID << "   ... loading antigen " << AntigenName << " from the library " << endl;
    std::pair<superProtein*, vector<int> > AG = getAntigen(AntigenName);

    // 2b - Preparing or loading the structures for this antigen
    // Note: we make each process check the file exists, in case problems of copying/file access between different machines
    // first, check if the structures are available (or the prepared compact interaction codes for this AA sequence):
    string fStruct = fnameStructures(AG.first, receptorSize, minInteract, AG.second);
    ifstream f(fStruct.c_str());
    if(f.good()){f.close();}
    else {
        string fCompact =   fileNameCompactForAASeqLigand(AG.first, receptorSize, minInteract, AG.second);
        ifstream f(fStruct.c_str());
        if(f.good()){f.close();}
        else {
            if(rankProcess == 0){
                cout << "\nERR: the list of binding structures for this antigen could not been found in this folder..." << endl;
                cout << "     its calculation can take typically 12 to 50 hours, so we do not recompute structures inside " << endl;
                cout << "     the 'repertoire' option, which is made to treat lots of sequences in multithreads, and the " << endl;
                cout << "     calculation of structures is not multithreaded, so it would waste resources." << endl;
                cout << "     => Please either find the structures file on the Absolut server, " << endl;
                cout << "     or run this program with the option 'singleBinding' and one CDR3 AA sequence, it will compute " << endl;
                cout << "     the structures and save them in the current folder." << endl;
                cout << "\n";
                cout << "     For information, the lacking file is:" << endl;
                cout << "     " << fStruct << endl;
                cout << "\n";
                cout << "     Or, alternately, the file with precomputed interaction codes:" << endl;
                cout << "     " << fStruct << endl;
                cout << "Bye!" << endl;
            }
            // each process will close
            return;
        }
    }



    // One process only makes an affinity calculation to regenerate the compact file if necessary
    #ifdef USE_MPI
    if(rankProcess == 0){
        cout << "The main process will check everything is ready to calculate affinities, and regenerate compact files if necessary" << endl;
        affinityOneLigand Ttest = affinityOneLigand(AG.first, receptorSize, minInteract, -1, 1, AG.second);
        Ttest.affinity(string(receptorSize+1, 'A'));
        cout << "   -> Everything ready!" << endl;
    }
    int res2 = MPI_Barrier(MPI_COMM_WORLD);
    if (res2 != MPI_SUCCESS) cout << myID << ", problem with MPI_Barrier inside option2()" << endl;
    #endif

    affinityOneLigand T3 = affinityOneLigand(AG.first, receptorSize, minInteract, -1, 1, AG.second);
    // This shortcuts a lots of documenting/debugging calculations, and only calculates best energy
    T3.setUltraFast(true);

    // 3 - Separating the job:
    //      - between the processes (1 ... nJobs) => startingLineThisJob ... endingLineThisJob (both included)
    //      - then between the threads insied each process => Will be done later
    int minLine = max(0, startingLine);
    int maxLine = min(static_cast<int>(rep.nLines())-1, endingLine);
    int nToProcess = maxLine - minLine + 1;

    int nPerJob = nToProcess / nJobs;   // number per MPI process

    int startingLineThisJob = minLine + rankProcess * nPerJob; // + 10000;
    int endingLineThisJob = minLine + (rankProcess + 1) * nPerJob - 1;
    if((endingLineThisJob - startingLineThisJob) > 1e6){
        cout << "WRN: Be prepared, very high number of sequences for this process. Might take forever...!\n" << endl;
    }
    if(rankProcess == nJobs - 1) endingLineThisJob = maxLine; // just for rounding it up
    cout << myID << "    ... will process lines " << startingLineThisJob << " to " << endingLineThisJob << " and then split into " << nThreads << " threads " << endl;


    dataset<binding>* commonSavingLocation = new dataset<binding>();
    commonSavingLocation->setNameAntigen(AntigenName);

    pthread_t tid[MAX_ALLOWED_THREADS]; // we allow max 50 thread
    for(size_t i = 0; i < static_cast<size_t>(nThreads); ++i){

        // cut into equal blocks between each thread
        int nToProcessThreads = endingLineThisJob - startingLineThisJob + 1;
        int nPerThread = nToProcessThreads / nThreads;
        int startingLineThisThread = startingLineThisJob + static_cast<int>(i) * nPerThread;
        int endingLineThisThread = startingLineThisJob + (static_cast<int>(i) + 1) * nPerThread - 1;
        if(static_cast<int>(i) == nThreads - 1) endingLineThisThread = endingLineThisJob;
        //cout << myID << "t" << i << "    ... will process lines " << startingLineThisThread << " to " << endingLineThisThread << "" << endl;

        stringstream resultFile;  resultFile << prefix << "TempBindingsFor" << AntigenName << "_t" << i << "_Part" << rankProcess+1 <<"_of_" << nJobs << ".txt";
        vector< std::pair<string, string> >* listToProcess = new vector< std::pair<string, string> >(rep.getLines(startingLineThisThread, endingLineThisThread));


        stringstream threadName; threadName << "Proc" << rankProcess << "_t" << i << "/" << nThreads;
        cout << threadName.str() << ", will treat " << listToProcess->size() << ", i.e. lines " << startingLineThisThread << " to " << endingLineThisThread << " included,  and save result in " << endl;
        cout << threadName.str() << ", " << resultFile.str() << endl;

        //struct argsThread *arguments = (struct argsThread *) malloc(sizeof(struct argsThread)); // this did segfault, C way of doing, bad bad bad
        argsThread *arguments = new argsThread();
        arguments->resultFile = resultFile.str();
        arguments->receptorSize = receptorSize;
        arguments->T3 = &T3;
        arguments->listToProcess = listToProcess;
        arguments->identificationThread = threadName.str();
        arguments->antigenName = ID_antigen;
        arguments->savingLocation = commonSavingLocation;

        if(nThreads > 1){
            int err = pthread_create(&(tid[i]), nullptr, &oneThreadJob, (void *) arguments);
            if (err != 0) cerr << "\ncan't create thread :" << strerror(err) << endl;
        } else {
            // in this case we don't need to start a thread, just call the function
            oneThreadJob((void*) arguments);
        }
    }

    if(nThreads > 1){
        cerr << myID << ", waiting for all threads to complete" << endl;
        for(size_t i = 0; i < static_cast<size_t>(nThreads); ++i){
            pthread_join(tid[i], nullptr);
            cout << myID << ", thread " << i << "/" << nThreads << " has finished " << endl;
        }
    }


    stringstream fName; fName << prefix << AntigenName << "FinalBindings_Process_" << rankProcess+1 << "_Of_" << nJobs << ".txt";
    cout << myID << "Saving pooled results into " << fName.str() << endl;
    commonSavingLocation->write(fName.str());



    // This mutex control the access to shared memory, can be cleared now
    pthread_mutex_destroy(&lockAccessPrecompAffinities);
    pthread_mutex_destroy(&lockSaveCommonDataset);

    cout << myID << ", =============== process finished! ==============" << endl;


    #ifdef USE_MPI
// This was a code to force MPI processes to wait for each-other using MPI barrier. Apparently, it is not necessary.
//    double centroid[3];/*ignore this array*/
//    if (rankProcess != 0) {
//        MPI_Recv(&centroid, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        // sleep(1); /*This represent many calculations that will happen here later, instead of sleep*/
//        cout << myID << "=> Will now wait for other process to complete " << endl;
//        int res = MPI_Barrier(MPI_COMM_WORLD);
//        if (res != MPI_SUCCESS) cout << myID << ", problem with MPI_Barrier" << endl;
//    } else {
//        for (int i=0; i<nJobs-1; i++)        {
//            MPI_Send(&centroid, 3, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);
//        }
//        int res = MPI_Barrier(MPI_COMM_WORLD);
//        if (res != MPI_SUCCESS) cout << myID << ", problem with MPI_Barrier" << endl;
//        cout << myID << " => All MPI processes have been completed! Main prosses will take over " << endl;
//    }

#ifdef WIN32
    Sleep(2);
    if(commonSavingLocation) delete commonSavingLocation;
    Sleep(2);
#else
    sleep(2);
    if(commonSavingLocation) delete commonSavingLocation;
    sleep(2);
#endif
    #endif
}