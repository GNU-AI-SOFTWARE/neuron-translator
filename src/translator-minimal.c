/*
 * Minimal Hurd translator for testing
 */

#include <stdio.h>
#include <string.h>

/* Hurd/Mach types */
typedef unsigned int mach_port_t;
#define MACH_PORT_NULL ((mach_port_t) 0)

struct mach_msg_header;
typedef struct mach_msg_header *mach_msg_header_t;

typedef int error_t;

struct iouser;
struct node;
struct iobuf;

/* Global variable */
char *fs_help = "Minimal Test Translator";

/* Minimal trivfs_demuxer */
int trivfs_demuxer(mach_msg_header_t inmsg, mach_msg_header_t outmsg)
{
    FILE *f = fopen("/home/claire/minimal_debug.log", "a");
    if (f) {
        fprintf(f, "[MINIMAL] trivfs_demuxer called!\n");
        fflush(f);
        fclose(f);
    }
    return 0;
}

/* Dummy fs_open */
error_t fs_open(struct iouser *cred, int flags, mode_t mode, struct node *node, struct iobuf **iobuf)
{
    FILE *f = fopen("/home/claire/minimal_debug.log", "a");
    if (f) {
        fprintf(f, "[MINIMAL] fs_open called!\n");
        fflush(f);
        fclose(f);
    }
    *iobuf = NULL;
    return 0;
}

/* Dummy fs_read */
error_t fs_read(struct iouser *cred, struct iobuf *iobuf, off_t offset, size_t *len, size_t count)
{
    FILE *f = fopen("/home/claire/minimal_debug.log", "a");
    if (f) {
        fprintf(f, "[MINIMAL] fs_read called!\n");
        fflush(f);
        fclose(f);
    }
    *len = 0;
    return 0;
}

/* Dummy fs_write */
error_t fs_write(struct iouser *cred, struct iobuf *iobuf, off_t offset, size_t len, size_t count)
{
    FILE *f = fopen("/home/claire/minimal_debug.log", "a");
    if (f) {
        fprintf(f, "[MINIMAL] fs_write called!\n");
        fflush(f);
        fclose(f);
    }
    return 0;
}
