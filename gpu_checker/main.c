#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#include <CL/cl.h>

#define CLEAR_LINE "\x1b[K"

#include "clutil.h"

#define TYPE CL_DEVICE_TYPE_ALL

static inline uint64_t get_timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t) ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}

static inline uint64_t start_timer() {
    return get_timer();
}

static inline uint64_t stop_timer(uint64_t start) {
    return get_timer() - start;
}

int main(int argc, char **argv) {
    FILE *log_file = fopen("out.txt", "w");
    if (!log_file) perror("Error opening logfile");
    const char *kernel_file = "kernel.cl";
    const char *kernel_name = "start";

    printf("PID: %u\n", getpid());

    cl_uint num_platforms;
    check(clGetPlatformIDs(0, NULL, &num_platforms), "getPlatformIDs (num)");
    printf("%d platforms:\n", num_platforms);

    cl_platform_id platforms[num_platforms];
    check(clGetPlatformIDs(num_platforms, platforms, NULL), "getPlatformIDs");

    cl_device_id device = NULL;
    cl_uint cus = 0;

    printf("Available platforms:\n");
    for (int i = 0; i < num_platforms; i++) {
        char *info = getPlatformInfo(platforms[i]);
        printf("%d: %s\n", i, info);
        free(info);
        cl_uint num_devices;
        check(clGetDeviceIDs(platforms[i], TYPE, 0, NULL, &num_devices), "getDeviceIDs (num)");

        cl_device_id devices[num_devices];
        check(clGetDeviceIDs(platforms[i], TYPE, num_devices, devices, NULL), "getDeviceIDs");

        printf("  %d available devices:\n", num_devices);
        for (int j = 0; j < num_devices; j++) {
            device_info *infos = getDeviceInfo(devices[j]);
            printf("    %s\n", infos->info_str);
            if (infos->compute_units >cus && (strstr(infos->info_str,"NVIDIA")||strstr(infos->info_str,"AMD"))) {
                cus = infos->compute_units;
                device = devices[j];
            }

        }
        putchar('\n');
    }

    if (!device) {
        fprintf(stderr, "No devices found.\n");
        return -1;
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    device_info *info = getDeviceInfo(device);
    printf("Using %s\n", info->info_str);
    fprintf(log_file, "Using %s\n", info->info_str);
    free(info);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
    cl_int error;
    const char *main_src = readFile(kernel_file);
    cl_program main_cl = clCreateProgramWithSource(context, 1, &main_src, NULL, &error);
    check(error, "Creating program");
    printf("Compiling...\n");
    error = clCompileProgram(main_cl, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL);
    if (error == CL_COMPILE_PROGRAM_FAILURE) {
        size_t log_size;
        clGetProgramBuildInfo(main_cl, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *) malloc(log_size);
        clGetProgramBuildInfo(main_cl, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s j\n", log);
        exit(error);
    } else if (error) {
        fprintf(stderr, "Error compiling: %s\n", getErrorString(error));
        exit(error);
    }
    const cl_program programs[] = {main_cl};
    printf("Linking...\n");
    cl_program program = clLinkProgram(context, 0, NULL, NULL, 1, programs, NULL, NULL, &error);
    check(error, "Linking");
    cl_kernel kernel = clCreateKernel(program, kernel_name, NULL);

    unsigned int x2total = 33;
    unsigned int x2step = 15;
    size_t ntotal = 1LLU << x2total;
    size_t nstep = 1LLU << min(x2step, x2total);
    uint64_t stride = 1LLU << (48U - x2total);

    printf("Number of spawned threads : %llu, at each step using : %llu threads, each thread going through stride: %llu. \n", ntotal, nstep, stride,ntotal*nstep*stride);
    fprintf(log_file, "Number of spawned threads : %llu, at each step using : %llu threads, each thread going through stride: %llu. \n", ntotal, nstep, stride,ntotal*nstep*stride);
    printf("Thus doing %lld steps for a total iteration of %lld. \n", ntotal/nstep,stride*ntotal);
    fprintf(log_file, "Thus doing %lld steps for a total iteration of %lld. \n", ntotal/nstep,stride*ntotal);
    fflush(stdout);
    cl_mem mem_seeds = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong) * nstep, NULL, NULL);
    cl_mem mem_debug = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong) * nstep, NULL, NULL);
    cl_mem mem_output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * nstep, NULL, NULL);

    check(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &stride), "Argument stride");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_seeds), "Argument seeds");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &mem_debug), "Argument debug");
    check(clSetKernelArg(kernel, 4, sizeof(cl_mem), &mem_output), "Argument output");

    size_t global_work_size[1] = {nstep};

    uint64_t *seeds = malloc(nstep * sizeof(uint64_t));
    uint64_t *debug = malloc(nstep * sizeof(uint64_t));
    uint32_t *output = malloc(nstep * sizeof(uint32_t));

    char *kf = malloc(strlen(kernel_file));
    strcpy(kf, kernel_file);
    kf[strlen(kf) - 3] = 0;
    printf("Executing %s.%s\n", kf, kernel_name);
    fprintf(log_file, "Executing %s.%s\n", kf, kernel_name);

	printf("Work Load possible: %d %d %d %d\n",CL_DEVICE_MAX_WORK_GROUP_SIZE,CL_DEVICE_MAX_WORK_ITEM_SIZES,CL_DEVICE_LOCAL_MEM_SIZE,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    fprintf(log_file, "Work Load possible: %d %d %d %d\n",CL_DEVICE_MAX_WORK_GROUP_SIZE,CL_DEVICE_MAX_WORK_ITEM_SIZES,CL_DEVICE_LOCAL_MEM_SIZE,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
    uint64_t t = start_timer();
    for (size_t offset = 0; offset < ntotal; offset += nstep) {
        for (size_t i = 0; i < nstep; i++) seeds[i] = UINT64_MAX;
        for (size_t i = 0; i < nstep; i++) debug[i] = UINT64_MAX;
        for (size_t i = 0; i < nstep; i++) output[i] = UINT32_MAX;
        float perc = offset * 100.f / ntotal;
        uint64_t t2 = start_timer();
        check(clSetKernelArg(kernel, 0, sizeof(offset), &offset), "Argument offset");
        printf("\rx  %3.3f%%", perc);
        fflush(stdout);
        check(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL), "\nExecute");
        check(clFinish(queue), "\nFinish execute");
        printf("\r<- %3.3f%%", perc);
        fflush(stdout);
        check(clEnqueueReadBuffer(queue, mem_output, 1, 0, nstep * sizeof(uint32_t), output, 0, NULL, NULL), "\nRead");
        check(clEnqueueReadBuffer(queue, mem_seeds, 1, 0, nstep * sizeof(uint64_t), seeds, 0, NULL, NULL), "\nRead");
        check(clEnqueueReadBuffer(queue, mem_debug, 1, 0, nstep * sizeof(uint64_t), debug, 0, NULL, NULL), "\nRead");
        uint64_t d2 = stop_timer(t2);
        printf("\rw  %3.3f%% %.4f" CLEAR_LINE, perc, d2 / 1000000.f);
        fflush(stdout);
        for (size_t i = 0; i < nstep; i++) {
            uint32_t count = output[i];
            //fprintf(log_file, "global id %04lld offset %04lld local counter %04lld out %04ld %04lld\n", seeds[i],offset,i,output[i],debug[i]);
            if (count != 0) {
                fprintf(log_file, "%016xlld %016xlld %08xld\n", seeds[i], debug[i],count);
                //printf("\n%lld %lld %ld\n", seeds[i],debug[i],count);
            }
        }
        fflush(log_file);
        uint64_t d = get_timer() - t;
        double per_item = (double) d / (offset + nstep);
        double eta = ((double) d / ((offset + nstep) * 1000000000.)) * (ntotal - offset - nstep);
        uint64_t d2ms = d2 / 1000000;
        printf("  %fs / %llu items = %lfns/item, %llums/batch, ETA: %lfs (%dh%dm%ds)", d / 1000000000.f, offset + nstep,
               per_item, d2ms, eta,
               (int) (eta / 3600), ((int) (eta / 60)) % 60, ((int) eta) % 60);

    }
    puts("\rDone.                                                       ");
    uint64_t d = stop_timer(t);
    fclose(log_file);

    printf("%fs / %llu items = %lfns/item\n", d / 1000000000.f, ntotal, (double) d / ntotal);
    // fclose(f);
    return 0;
}

