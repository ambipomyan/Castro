
#include <new>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#ifndef WIN32
#include <unistd.h>
#endif

#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_AmrLevel.H>

#include <ctime>

#include <Castro.H>
#include <Castro_io.H>

#include <global.H>

using namespace castro;

using namespace amrex;

std::string inputs_name{};

amrex::LevelBld* getLevelBld ();

// Any parameters we want to override the defaults for in AMReX

void override_parameters ()
{
    {
        ParmParse pp("amrex");
#ifndef RADIATION // Radiation is not yet ready to stop using managed memory
        if (!pp.contains("the_arena_is_managed")) {
            // Use device memory allocations, not managed memory.
            pp.add("the_arena_is_managed", false);
        }
#endif
        if (!pp.contains("abort_on_out_of_gpu_memory")) {
            // Abort if we run out of GPU memory.
            pp.add("abort_on_out_of_gpu_memory", true);
        }
    }

    {
        ParmParse pp("amr");
        // Always check for whether to dump a plotfile or checkpoint.
        if (!pp.contains("message_int")) {
            pp.add("message_int", 1);
        }
    }
}

int
main (int   argc,
      char* argv[])
{

    // check to see if it contains --describe
    if (argc >= 2) {
        for (auto i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--describe") {
                Castro::writeBuildInfo();
                return 0;
            }
        }
    }

    //
    // Make sure to catch new failures.
    //
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, override_parameters);
    {

    // Refuse to continue if we did not provide an inputs file.

    if (argc <= 1) {
        amrex::Abort("Error: no inputs file provided on command line.");
    }

    // Save the inputs file name for later.

    if (!strchr(argv[1], '=')) {
        inputs_name = argv[1];
    }

    BL_PROFILE_VAR("main()", pmain);

    double dRunTime1 = ParallelDescriptor::second();

    std::cout << std::setprecision(10);

    int  max_step;
    Real strt_time;
    Real stop_time;
    ParmParse pp;

    max_step  = -1;
    strt_time =  0.0;
    stop_time = -1.0;

    pp.query("max_step",max_step);
    pp.query("strt_time",strt_time);
    pp.query("stop_time",stop_time);

    if (strt_time < 0.0)
    {
        amrex::Abort("MUST SPECIFY a non-negative strt_time");
    }

    if (max_step < 0 && stop_time < 0.0) {
      amrex::Abort(
       "Exiting because neither max_step nor stop_time is non-negative.");
    }

    // Print the current date and time.

    time_t time_type;

    struct tm* time_pointer = nullptr;

    time(&time_type);

    time_pointer = gmtime(&time_type);

    amrex::Print() << std::setfill('0') << "\nStarting run at "
                   << std::setw(2) << time_pointer->tm_hour << ":"
                   << std::setw(2) << time_pointer->tm_min << ":"
                   << std::setw(2) << time_pointer->tm_sec << " UTC on "
                   << time_pointer->tm_year + 1900 << "-"
                   << std::setw(2) << time_pointer->tm_mon + 1 << "-"
                   << std::setw(2) << time_pointer->tm_mday << "." << std::endl;

    //
    // Initialize random seed after we're running in parallel.
    //

    Amr* amrptr = new Amr(getLevelBld());
    global::the_amr_ptr = amrptr;

    amrptr->init(strt_time,stop_time);

    // If we set the regrid_on_restart flag and if we are *not* going to take
    //    a time step then we want to go ahead and regrid here.
    if ( amrptr->RegridOnRestart() &&
         ( (amrptr->levelSteps(0) >= max_step) ||
           (amrptr->cumTime() >= stop_time) ) )
           {
           //
           // Regrid only!
           //
           amrptr->RegridOnly(amrptr->cumTime());
           }

    double dRunTime2 = ParallelDescriptor::second();

    // file to save results
    std::fstream f;
    f.open("out.txt");
    f<<"header"<<std::endl;
    // counts
    int counts = 0;
    // timer
    Real prevTime[4];
    for (int i = 0; i < 4; i++) {
        prevTime[i] = 0.0;
    }
    // diagnostic variables
    Real temp_2 = 0.0;
    Real mom_2 = 0.0;
    Real mass_2 = 0.0;
    Real energy_2 = 0.0;

    // controls
    //stop_time = 0.25*stop_time;
    Real train_partition = 1.0;
    int train_iters = 10000;
    Real train_lr = 0.0001;

    int model_size = 5;
    int data_size = model_size + 1;

    Real *X = (Real *)malloc(data_size*sizeof(Real));
    X[0] = 1.0;
    for (int i = 1; i < data_size; i++) {
        X[i] = 0.0;
    }
    Real y = 0.0;
    Real *a = (Real *)malloc(model_size*sizeof(Real));
    for (int i = 0; i < model_size; i++) {
	a[i] = 0.2;
    }

    Real res_time_step = 0.0;
    int res_iter = 0;


    while ( amrptr->okToContinue()                            &&
           (amrptr->levelSteps(0) < max_step || max_step < 0) &&
           (amrptr->cumTime() < stop_time || stop_time < 0.0) )

    {
        //
        // Do a timestep.
        //
        amrptr->coarseTimeStep(stop_time);

	// td_region_end()
	counts++;

	f<<counts<<" ";

        // finest level
        int finest_level = amrptr->finestLevel();

	// levels
	amrex::Vector< std::unique_ptr<AmrLevel> >& amr_levels = amrptr->getAmrLevels();

	// collecting data from all levels
	Real temp = 0.0;
	Real mom = 0.0;
	Real mass = 0.0;
	Real energy = 0.0;

        for (int lev = 0; lev <= finest_level; lev++) {
	    Real dt = amrptr->dtLevel(lev);
	    Real prev_time = prevTime[lev];
            
	    // dummy object
	    Castro& ca_lev = dynamic_cast<Castro &>(amrptr->getLevel(lev));

            amrex::MultiFab& S_new = amr_levels[lev]->get_new_data(State_Type);

            temp = ca_lev.volWgtSum(S_new, UTEMP);
            mom = ca_lev.volWgtSum(S_new, UMX);
#ifdef HYBRID_MOMENTUM
	    mom = ca_lev.volWgtSum(S_new, UML);
#endif
	    mass  += ca_lev.volWgtSum(S_new, URHO);
            energy += ca_lev.volWgtSum(S_new, UEDEN);

            prev_time += dt;
	    prevTime[lev] = prev_time;
        }

	// normalization
        temp = temp / 1e38;
        mom = mom / 1e51;

	Real dTemp = temp - temp_2;
	Real dMom = mom - mom_2;

        temp_2 = temp;
	mom_2 = mom;

        mass = mass / 1e33;
        energy = energy / 1e50;

	Real dMass = mass - mass_2;
	Real dEnergy = energy - energy_2;

	mass_2 = mass;
        energy_2 = energy;

        f<<temp<<" "<<dTemp<<" "<<mom<<" "<<dMom<<"  ";

	f<<mass<<" "<<dMass<<" "<<energy<<" "<<dEnergy<<" "<<prevTime[0];

/* batch data collection */

	Real train_var = temp;

	if (counts < data_size) {
	    int Idx = counts%data_size;
		X[Idx] = train_var;
	} else {
	    for (int i = 1; i < model_size; i++) {
                X[i] = X[i+1];
            }
	    X[model_size] = train_var;
	    y = train_var;
	}

/* varibale tracking */
/*
	if (counts > model_size) {
	    Real tar1 = (X[model_size] - X[model_size-1]) - (X[model_size-1] - X[model_size-2]);
	    Real tar2 = (X[2] - X[1]) - (X[1] -X[0]);

	    if (tar1 * tar2 < 0.0) {
	        res_time_step = prevTime[0];
		res_iter = counts;
	    }
	}
 */
/* simulation prediction */
/*
        if (amrptr->cumTime() < train_partition*stop_time &&
	    counts > model_size)
	{

	    Real *temp_step = (Real *)malloc(model_size*sizeof(Real));

            for (int k = 0; k < train_iters; k++) {
                for (int i = 0; i < model_size; i++) {
                    Real train_loss = 0.0;
                    Real hx = 0.0;
                    for (int j  = 0; j < model_size; j++) {
                        hx += a[j] * X[j];
                    }
                    train_loss += (hx - y) * X[i];
                    temp_step[i] = train_loss * train_lr;
                    a[i] -= temp_step[i];
	        }
	    }
	}

	Real train_pred = 0.0;
	Real train_error = 0.0;

	for (int i = 0; i < model_size; i++) {
            train_pred += a[i] * X[i];
	    train_error += train_pred - train_var; 
        }

	f<<" "<<train_pred<<" "<<train_var<<" "<<train_error<<" "<<res_time_step<<" "<<res_iter<<" "<<X[0]<<" "<<X[1]<<" "<<X[2]<<" "<<X[3]<<" "<<X[4]<<" "<<y<<"  "<<a[0]<<" "<<a[1]<<" "<<a[2]<<" "<<a[3]<<" "<<a[4];
 */
	printf("\n");

	f<<std::endl;

    }

    f.close();
    //

#ifdef DO_PROBLEM_POST_SIMULATION
    Castro::problem_post_simulation(amrptr->getAmrLevels());
#endif

    // Write final checkpoint and plotfile

    if (Castro::get_output_at_completion() == 1) {

        if (amrptr->stepOfLastCheckPoint() < amrptr->levelSteps(0)) {
            amrptr->checkPoint();
        }

        if (amrptr->stepOfLastPlotFile() < amrptr->levelSteps(0)) {
            amrptr->writePlotFile();
        }

        if (amrptr->stepOfLastSmallPlotFile() < amrptr->levelSteps(0)) {
            amrptr->writeSmallPlotFile();
        }

    }

    // Start calculating the figure of merit for this run: average number of zones
    // advanced per microsecond. This must be done before we delete the Amr
    // object because we need to scale it by the number of zones on the coarse grid.

    long numPtsCoarseGrid = amrptr->getLevel(0).boxArray().numPts();
    Real fom = Castro::num_zones_advanced * static_cast<Real>(numPtsCoarseGrid);

    time(&time_type);

    time_pointer = gmtime(&time_type);

    amrex::Print() << std::setfill('0') << "\nEnding run at "
                   << std::setw(2) << time_pointer->tm_hour << ":"
                   << std::setw(2) << time_pointer->tm_min << ":"
                   << std::setw(2) << time_pointer->tm_sec << " UTC on "
                   << time_pointer->tm_year + 1900 << "-"
                   << std::setw(2) << time_pointer->tm_mon + 1 << "-"
                   << std::setw(2) << time_pointer->tm_mday << "." << std::endl;

    delete amrptr;
    //
    // This MUST follow the above delete as ~Amr() may dump files to disk.
    //
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    const int nprocs = ParallelDescriptor::NProcs();

    double dRunTime3 = ParallelDescriptor::second();

    Real runtime_total = static_cast<Real>(dRunTime3 - dRunTime1);
    Real runtime_timestep = static_cast<Real>(dRunTime3 - dRunTime2);

    ParallelDescriptor::ReduceRealMax(runtime_total,IOProc);
    ParallelDescriptor::ReduceRealMax(runtime_timestep,IOProc);

    if (ParallelDescriptor::IOProcessor())
    {
        std::cout << "Run time = " << runtime_total << std::endl;
        std::cout << "Run time without initialization = " << runtime_timestep << std::endl;

        fom = fom / runtime_timestep / 1.e6;

        std::cout << "\n";
        std::cout << "  Average number of zones advanced per microsecond: " << std::fixed << std::setprecision(3) << fom << "\n";
        std::cout << "  Average number of zones advanced per microsecond per rank: " << std::fixed << std::setprecision(3) << fom / nprocs << "\n";
        std::cout << "\n";
    }

    if (auto* arena = dynamic_cast<CArena*>(amrex::The_Arena()))
    {
        //
        // A barrier to make sure our output follows that of RunStats.
        //
        ParallelDescriptor::Barrier();
        //
        // We're using a CArena -- output some FAB memory stats.
        //
        // This'll output total # of bytes of heap space in the Arena.
        //
        // It's actually the high water mark of heap space required by FABs.
        //
        char buf[256];

        sprintf(buf,
                "CPU(%d): Heap Space (bytes) used by Coalescing FAB Arena: %ld",
                ParallelDescriptor::MyProc(),
                arena->heap_space_used());

        std::cout << buf << std::endl;
    }

    BL_PROFILE_VAR_STOP(pmain);
    BL_PROFILE_SET_RUN_TIME(dRunTime2);

    }
    amrex::Finalize();

    return 0;
}
