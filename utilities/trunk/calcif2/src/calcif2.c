/***************************************************************************
 *   Copyright (C) 2008-2019 by Walter Brisken & Adam Deller               *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
//===========================================================================
// SVN properties (DO NOT CHANGE)
//
// $Id$
// $HeadURL: $
// $LastChangedRevision$
// $Author$
// $LastChangedDate$
//
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <fcntl.h>
#include <rpc/rpc.h>
#include <glob.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "config.h"
#include "difxcalc.h"
#include "CALCServer.h"

#define MAX_FILES	2048

const char program[] = "calcif2";
const char author[]  = "Walter Brisken <wbrisken@nrao.edu>";
const char version[] = VERSION;
const char verdate[] = "20190401";

typedef struct
{
	int verbose;
	int force;
	int doall;
	double delta;	/* derivative step size, radians. <0 for noaber */
	char calcServer[DIFXIO_NAME_LENGTH];
	int calcProgram;
	int calcVersion;
	int nFile;
	int polyOrder;
	int polyInterval;	/* (sec) */
	int polyOversamp;
	int interpol;
	int allowNegDelay;
	char *files[MAX_FILES];
	int overrideVersion;
	enum AberCorr aberCorr;
} CommandLineOptions;

static void usage()
{
	fprintf(stderr, "%s ver. %s  %s  %s\n\n", program, version, author, verdate);
	fprintf(stderr, "A program to calculate a model for DiFX using a calc server.\n\n");
	fprintf(stderr, "Usage : %s [options] { <calc file> | -a }\n\n", program);
	fprintf(stderr, "<calc file> should be a '.calc' file as generated by vex2difx.\n\n");
	fprintf(stderr, "options can include:\n");
	fprintf(stderr, "  --help\n");
	fprintf(stderr, "  -h                      Print this help and quit\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --verbose\n");
	fprintf(stderr, "  -v                      Be more verbose in operation\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --quiet\n");
	fprintf(stderr, "  -q                      Be less verbose in operation\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --force\n");
	fprintf(stderr, "  -f                      Force recalc\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --noaber\n");
	fprintf(stderr, "  -n                      Don't do aberration, etc, corrections\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --noatmos\n");
	fprintf(stderr, "  -A                      Don't include atmosphere in UVW calculations\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --all\n");
	fprintf(stderr, "  -a                      Do all calc files found\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --allow-neg-delay\n");
	fprintf(stderr, "  -z                      Don't zero negative delays\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --order <n>\n");
	fprintf(stderr, "  -o      <n>             Use <n>th order polynomial [5]\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --oversamp <m>\n");
	fprintf(stderr, "  -O         <m>          Oversample polynomial by factor <m> [1]\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --interval <int>\n");
	fprintf(stderr, "  -i         <int>        New delay poly every <int> sec. [120]\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --fit\n");
	fprintf(stderr, "  -F                      Fit oversampled polynomials\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --override-version      Ignore difx versions\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "  --server <servername>\n");
	fprintf(stderr, "  -s       <servername>   Use <servername> as calcserver\n\n");
	fprintf(stderr, "      By default 'localhost' will be the calcserver.  An environment\n");
	fprintf(stderr, "      variable CALC_SERVER can be used to override that.  The command line\n");
	fprintf(stderr, "      overrides all.\n");
	fprintf(stderr, "\n");
}

static void deleteCommandLineOptions(CommandLineOptions *opts)
{
	int i;

	if(!opts)
	{
		return;
	}

	for(i = 0; i < opts->nFile; ++i)
	{
		free(opts->files[i]);
	}

	free(opts);
}

static CommandLineOptions *newCommandLineOptions(int argc, char **argv)
{
	CommandLineOptions *opts;
	glob_t globbuf;
	int i, v;
	char *cs;
	int die = 0;

	opts = (CommandLineOptions *)calloc(1, sizeof(CommandLineOptions));
	opts->delta = 0.0001;
	opts->polyOrder = 5;
	opts->polyOversamp = 1;
	opts->polyInterval = 120;
	opts->interpol = 0;	/* usual solve */
	opts->aberCorr = AberCorrExact;

	for(i = 1; i < argc; ++i)
	{
		if(argv[i][0] == '-')
		{
			if(strcmp(argv[i], "-v") == 0 ||
			   strcmp(argv[i], "--verbose") == 0)
			{
				++opts->verbose;
			}
			else if(strcmp(argv[i], "-q") == 0 ||
			   strcmp(argv[i], "--quiet") == 0)
			{
				--opts->verbose;
			}
			else if(strcmp(argv[i], "-f") == 0 ||
				strcmp(argv[i], "--force") == 0)
			{
				++opts->force;
			}
			else if(strcmp(argv[i], "-a") == 0 ||
			        strcmp(argv[i], "--all") == 0)
			{
				opts->doall = 1;
			}
			else if(strcmp(argv[i], "-z") == 0 ||
			        strcmp(argv[i], "--allow-neg-delay") == 0)
			{
				opts->allowNegDelay = 1;
			}
			else if(strcmp(argv[i], "-n") == 0 ||
				strcmp(argv[i], "--noaber") == 0)
			{
				opts->aberCorr = AberCorrUncorrected;
				opts->delta = -1.0;
			}
			else if(strcmp(argv[i], "-A") == 0 ||
				strcmp(argv[i], "--noatmos") == 0)
			{
				opts->aberCorr = AberCorrNoAtmos;
			}
			else if(strcmp(argv[i], "-F") == 0 ||
				strcmp(argv[i], "--fit") == 0)
			{
				opts->interpol = 1;
			}
			else if(strcmp(argv[i], "-h") == 0 ||
				strcmp(argv[i], "--help") == 0)
			{
				usage();
				deleteCommandLineOptions(opts);
				
				return 0;
			}
			else if(strcmp(argv[i], "--override-version") == 0)
			{
				opts->overrideVersion = 1;
			}
			else if(i+1 < argc)
			{
				if(strcmp(argv[i], "--server") == 0 ||
				   strcmp(argv[i], "-s") == 0)
				{
					++i;
					v = snprintf(opts->calcServer, DIFXIO_NAME_LENGTH, "%s", argv[i]);
					if(v >= DIFXIO_NAME_LENGTH)
					{
						fprintf(stderr, "Error: calcif2: calcServer name, %s, is too long (more than %d chars)\n",
							argv[i], DIFXIO_NAME_LENGTH-1);
						++die;
					}
				}
				else if(strcmp(argv[i], "--order") == 0 ||
					strcmp(argv[i], "-o") == 0)
				{
					++i;
					opts->polyOrder = atoi(argv[i]);
				}
				else if(strcmp(argv[i], "--oversamp") == 0 ||
					strcmp(argv[i], "-O") == 0)
				{
					++i;
					opts->polyOversamp = atoi(argv[i]);
				}
				else if(strcmp(argv[i], "--interval") == 0 ||
					strcmp(argv[i], "-i") == 0)
				{
					++i;
					opts->polyInterval = atoi(argv[i]);
				}
				else if(argv[i][0] == '-')
				{
					printf("Error: calcif2: Illegal option : %s\n", argv[i]);
					++die;
				}
			}
			else if(argv[i][0] == '-')
			{
				printf("Error: calcif2: Illegal option : %s\n", argv[i]);
				++die;
			}
		}
		else
		{
			opts->files[opts->nFile] = strdup(argv[i]);
			opts->nFile++;
			if(opts->nFile >= MAX_FILES)
			{
				fprintf(stderr, "Error: calcif2: Too many files (%d max)\n", MAX_FILES);
				++die;
			}
		}
	}

	if(opts->doall == 0 && opts->nFile == 0 && !die)
	{
		fprintf(stderr, "Error: calcif2: No input files!\n");
		++die;
	}

	if(opts->polyOrder < 2 || opts->polyOrder > MAX_MODEL_ORDER)
	{
		fprintf(stderr, "Error: calcif2 Polynomial order must be in range [2, %d]\n", MAX_MODEL_ORDER);
		++die;
	}

	if(opts->polyOversamp < 1 || opts->polyOversamp > MAX_MODEL_OVERSAMP)
	{
		fprintf(stderr, "Error: calcif2 Polynomial oversample factor must be in range [1, %d]\n", MAX_MODEL_OVERSAMP);
		++die;
	}

	if(opts->interpol == 1 && opts->polyOversamp == 1)
	{
		opts->polyOversamp = 2;
		fprintf(stderr, "Note: oversampling increased to 2 because polynomial fitting is being used.\n");
	}

	if(opts->polyInterval < 10 || opts->polyInterval > 600)
	{
		fprintf(stderr, "Error: calcif2: Interval must be in range [10, 600] sec\n");
		++die;
	}

	if(opts->nFile > 0 && opts->doall)
	{
		fprintf(stderr, "Error: calcif2: Option '--all' provided with files!\n");
		++die;
	}
	else if(opts->doall > 0)
	{
		glob("*.calc", 0, 0, &globbuf);
		opts->nFile = globbuf.gl_pathc;
		if(opts->nFile >= MAX_FILES)
		{
			fprintf(stderr, "Error: calcif2: Too many files (%d max)\n", MAX_FILES);
			++die;
		}
		else if(opts->nFile <= 0)
		{
			fprintf(stderr, "Error: calcif2: No .calc files found.  Hint: Did you run vex2difx yet???\n");
			++die;
		}
		for(i = 0; i < opts->nFile; ++i)
		{
			opts->files[i] = strdup(globbuf.gl_pathv[i]);
		}
		globfree(&globbuf);
	}

	if(opts->calcServer[0] == 0)
	{
		cs = getenv("CALC_SERVER");
		if(cs)
		{
			v = snprintf(opts->calcServer, DIFXIO_NAME_LENGTH, "%s", cs ? cs : "localhost");
			if(v >= DIFXIO_NAME_LENGTH)
			{
				fprintf(stderr, "Error: env var CALC_SERVER is set to a name that is too long, %s (should be < 32 chars)\n", cs ? cs : "localhost");
				++die;
			}
		}
	}

	opts->calcVersion = CALCVERS;
	opts->calcProgram = CALCPROG;

	if(die)
	{
		if(die > 1)
		{
			fprintf(stderr, "calcif2 quitting. (%d errors)\n", die);
		}
		else
		{
			fprintf(stderr, "calcif2 quitting.\n");
		}
		fprintf(stderr, "Use -h option for calcif2 help.\n");
		deleteCommandLineOptions(opts);

		return 0;
	}

	return opts;
}

/* return 1 if f2 exists and is older than f1 */
static int skipFile(const char *f1, const char *f2)
{
	struct stat s1, s2;
	int r1, r2;

	r2 = stat(f2, &s2);
	if(r2 != 0)
	{
		return 0;
	}
	r1 = stat(f1, &s1);
	if(r1 != 0)
	{
		return 0;
	}

	if(s2.st_mtime > s1.st_mtime)
	{
		return 1;
	}

	return 0;
}

static void tweakDelays(DifxInput *D, const char *tweakFile, int verbose)
{
	const int MaxLineSize=100;
	FILE *in;
	char line[MaxLineSize];
	int s, a, i, j;
	double mjd, A, B, C;
	DifxPolyModel ***im, *model;
	int nModified = 0;
	int nModel = 0;
	int nLine;
	char *v;

	in = fopen(tweakFile, "r");
	if(!in)
	{
		/* The usual case. */

		return;
	}

	printf("Delay tweaking file %s found!\n", tweakFile);

	for(nLine = 0; ; ++nLine)
	{
		v = fgets(line, MaxLineSize-1, in);
		if(feof(in) || v == 0)
		{
			break;
		}
		if(sscanf(line, "%lf %lf %lf %lf", &mjd, &A, &B, &C) != 4)
		{
			continue;
		}

		for(s = 0; s < D->nScan; ++s)
		{
			im = D->scan[s].im;
			if(!im)
			{
				continue;
			}
			for(a = 0; a < D->nAntenna; ++a)
			{
				if(!im[a])
				{
					continue;
				}
				for(i = 0; i <= D->scan[s].nPhaseCentres; ++i)
				{
					if(!im[a][i])
					{
						continue;
					}
					for(j = 0; j < D->scan[s].nPoly; ++j)
					{
						model = im[a][i] + j;
						if(fabs(model->mjd + model->sec/86400.0 - mjd) < 0.5/86400.0)	/* a match! */
						{
							++nModified;
							if(verbose > 1)
							{
								printf("Match found: ant=%d mjd=%d sec=%d = %15.9f\n", a, model->mjd, model->sec, mjd);
							}
							model->delay[0] += A;
							if(model->order > 0)
							{
								model->delay[1] += B;
							}
							if(model->order > 1)
							{
								model->delay[2] += C;
							}
						}
						if(nLine == 0)
						{
							++nModel;
						}
					}
				}
			}
		}
	}

	if(nModified != nModel)
	{
		printf("WARNING: calcif2: Only %d of %d models modified!\n", nModified, nModel);
	}
	else
	{
		printf("calcif2: All %d models modified.\n", nModel);
	}

	fclose(in);
}

void deleteCalcParams(CalcParams *p)
{
	if(p->clnt)
	{
		clnt_destroy(p->clnt);
	}
	free(p);
}

CalcParams *newCalcParams(const CommandLineOptions *opts)
{
	CalcParams *p;

	p = (CalcParams *)calloc(1, sizeof(CalcParams));

	p->increment = opts->polyInterval;
	p->order = opts->polyOrder;
	p->oversamp = opts->polyOversamp;
	p->delta = opts->delta;
	p->interpol = opts->interpol;
	p->aberCorr = opts->aberCorr;

	/* We know that opts->calcServer is no more than DIFXIO_NAME_LENGTH-1 chars long */
	strcpy(p->calcServer, opts->calcServer);
	p->calcProgram = opts->calcProgram;
	p->calcVersion = opts->calcVersion;
	p->allowNegDelay = opts->allowNegDelay;

	p->clnt = clnt_create(p->calcServer, p->calcProgram, p->calcVersion, "tcp");
	if(!p->clnt)
	{
		clnt_pcreateerror(p->calcServer);
		fprintf(stderr, "Error: calcif2: RPC clnt_create fails for host : %-s\n", p->calcServer);
		deleteCalcParams(p);

		return 0;
	}
	if(opts->verbose > 1)
	{
		printf("RPC client created\n");
	}

	return p;
}

static int runfile(const char *prefix, const CommandLineOptions *opts, CalcParams **p)
{
	DifxInput *D;
	FILE *in;
	char fn[DIFXIO_FILENAME_LENGTH];
	int v;
	const char *difxVersion;
	char delayModel[DIFXIO_FILENAME_LENGTH] = "";
	int doVMF = 0;	/* after running, replace atmosphere with Vienna Mapping Function (VMF) version */
	int doMet = 0;	/* after running, replace atmosphere with VMF and use supplied weather data */

	difxVersion = getenv("DIFX_VERSION");

	v = snprintf(fn, DIFXIO_FILENAME_LENGTH, "%s.calc", prefix);
	if(v >= DIFXIO_FILENAME_LENGTH)
	{
		fprintf(stderr, "Error: filename %s.calc is too long (max %d chars)\n", prefix, DIFXIO_FILENAME_LENGTH-1);
	}
	in = fopen(fn, "r");
	if(!in)
	{
		fprintf(stderr, "File %s not found or cannot be opened.  Quitting.\n", fn);

		return -1;
	}
	else
	{
		fclose(in);
	}

	D = loadDifxCalc(prefix);

	if(D == 0)
	{
		fprintf(stderr, "Error: loadDifxCalc(\"%s\") returned 0\n", prefix);

		return -1;
	}

	D = updateDifxInput(D, 0);
	if(D == 0)
	{
		fprintf(stderr, "Error: updateDifxInput(\"%s\") returned 0\n", prefix);

		return -1;
	}

	if(opts->force == 0 && skipFile(D->job->calcFile, D->job->imFile))
	{
		printf("Skipping %s due to file ages.\n", prefix);
		deleteDifxInput(D);

		return 0;
	}
	
	if(difxVersion && D->job->difxVersion[0])
	{
		if(strncmp(difxVersion, D->job->difxVersion, DIFXIO_VERSION_LENGTH-1))
		{
			printf("Attempting to run calcif2 from version %s on a job make for version %s\n", difxVersion, D->job->difxVersion);
			if(opts->overrideVersion)
			{
				fprintf(stderr, "Continuing because of --override-version\n");
			}
			else
			{
				fprintf(stderr, "calcif2 won't run on mismatched version without --override-version.\n");
				deleteDifxInput(D);

				return -1;
			}
		}
	}
	else if(!D->job->difxVersion[0])
	{
		printf("Warning: calcif2: working on unversioned job\n");
	}

	if(strlen(D->job->delayModel) > 0)
	{
		if(strcasecmp(D->job->delayModel, "VMF") == 0)
		{
			doVMF = 1;
		}
		else if(strcasecmp(D->job->delayModel, "Met") == 0)
		{
			doVMF = 1;
			doMet = 1;
		}
		else
		{
			int i;

			snprintf(delayModel, DIFXIO_FILENAME_LENGTH, "%s", D->job->delayModel);
			for(i = 0; delayModel[i]; ++i)
			{
				if(delayModel[i] == '+')
				{
					delayModel[i] = 0;
					if(strcasecmp(delayModel+i+1, "VMF") == 0)
					{
						doVMF = 1;
					}
					else if(strcasecmp(delayModel+i+1, "Met") == 0)
					{
						doVMF = 1;
						doMet = 1;
					}
					break;
				}
			}
		}
	}

	if(strlen(delayModel) > 0)
	{
		/* use specified delay model program rather than the calcserver */
		
		const int MaxCommandLength = 1024;
		char cmd[MaxCommandLength];

		if(opts->verbose > 1)
		{
			printDifxInput(D);
		}

		snprintf(cmd, MaxCommandLength, "%s %s.calc", delayModel, prefix);
		if(opts->verbose > 0)
		{
			printf("Executing the following: %s\n", cmd);
		}
		system(cmd);
		if(opts->verbose > 0)
		{
			printf("Done.\n");
		}
	}
	else
	{
		/* we know opts->calcServer is no more than DIFXIO_NAME_LENGTH-1 chars long */
		strcpy(D->job->calcServer, opts->calcServer);
		D->job->calcProgram = opts->calcProgram;
		D->job->calcVersion = opts->calcVersion;

		if(*p == 0)
		{
			/* make calcserver client */

			*p = newCalcParams(opts);
			if(*p == 0)
			{
				fprintf(stderr, "Error: Cannot initialize CalcParams\n");

				exit(EXIT_FAILURE);
			}
		}

		if(opts->verbose > 1)
		{
			printDifxInput(D);
		}

		v = difxCalcInit(D, *p);
		if(v < 0)
		{
			deleteDifxInput(D);
			fprintf(stderr, "Error: calcif2: difxCalcInit returned %d\n", v);

			return -1;
		}
		v = difxCalc(D, *p, prefix, opts->verbose);
		if(v < 0)
		{
			deleteDifxInput(D);
			fprintf(stderr, "Error: calcif2: difxCalc returned %d\n", v);

			return -1;
		}
		if(opts->verbose > 0)
		{
			printf("About to write IM file: %s\n", D->job->imFile);
		}
		tweakDelays(D, "calcif2.delay", opts->verbose);
		writeDifxIM(D);
		if(opts->verbose > 0)
		{
			printf("Wrote IM file\n");
		}
	}
	deleteDifxInput(D);

	if(doVMF)
	{
		const int MaxCmdLength=1024;
		char cmd[MaxCmdLength];

		snprintf(cmd, MaxCmdLength, "difxvmf %s %s\n", (doMet ? "--usewx" : ""), prefix);
		if(opts->verbose > 0)
		{
			printf("About to push VMF%s into IM file: %s\n", (doMet ? "+WX" : ""), D->job->imFile);
			if(opts->verbose > 1)
			{
				printf("  Executing: %s\n", cmd);
			}
		}
		system(cmd);
	}

	return 0;
}

int run(const CommandLineOptions *opts)
{
	CalcParams *p = 0;
	int i, l;

	if(getenv("DIFX_GROUP_ID"))
	{
		umask(2);
	}

	if(opts == 0)
	{
		return EXIT_FAILURE;
	}
		
	for(i = 0; i < opts->nFile; ++i)
	{
		l = strlen(opts->files[i]);
		if(l > 6)
		{
			if(strcmp(opts->files[i]+l-6, ".input") == 0)
			{
				opts->files[i][l-6] = 0;
			}
			else if(strcmp(opts->files[i]+l-5, ".calc") == 0)
			{
				opts->files[i][l-5] = 0;
			}
		}
		if(opts->verbose >= 0)
		{
			printf("%s processing file %d/%d = %s\n", program, i+1, opts->nFile, opts->files[i]);
		}
		runfile(opts->files[i], opts, &p);
	}
	if(p)
	{
		deleteCalcParams(p);
	}

	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
        int status;
	CommandLineOptions *opts;

	opts = newCommandLineOptions(argc, argv);

	status = run(opts);

	deleteCommandLineOptions(opts);
	
	return status;
}
