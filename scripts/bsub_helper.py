"""
Gregory Way 2017
Variational Autoencoder - Pan Cancer
scripts/bsub_helper.py

Usage: Run in command line with required command argument:

        python scripts/bsub_helper.py --command <command string>

There are also optional arguments:

    --queue             string of which queue to submit to
                            default: gpu
    --num_gpus          greater than the number of gpus to request per node
                            default: 0
    --num_gpus_shared   how many gpus are alotted to be shared
                            default: False if flag omitted
    --wall_time         time to spend on wall, in format hour:minute
                            default: 0:10 (ten minutes)
    --error_file        file name of where to write standard error
                            default: std_error.txt
    --output_file       file name of where to write standard output
                            default: std_output.txt

Output:
Will submit a job to the PMACS cluster queue
"""


class bsub_help():
    def __init__(self, command, queue='gpu', num_gpus=2, num_gpus_shared=0,
                 walltime='0:10', error_file='std_err.txt',
                 output_file='std_out'):
        try:
            self.command = command.split(' ')
        except:
            self.command = command
        self.queue = queue
        self.num_gpus = num_gpus
        self.num_gpus_shared = num_gpus_shared
        self.walltime = walltime
        self.error_file = error_file
        self.output_file = output_file

    def make_command_list(self):
        command_list = ['bsub', '-q', self.queue, '-eo', self.error_file,
                        '-oo', self.output_file, '-c', self.walltime]
        if self.queue == 'gpu':
            command_list += ['-R',
                             '"select[ngpus>{}] rusage [ngpus_shared={}]"'
                             .format(self.num_gpus, self.num_gpus_shared)]
        command_list += self.command
        return command_list

    def make_command_string(self):
        command_string = (
            'bsub -q {} -eo {} -oo {} -c {}'
            .format(self.queue, self.error_file, self.output_file,
                    self.walltime)
            )
        if self.queue == 'gpu':
            command_string = (
                '{} [-R "select[ngpus>{}] rusage [ngpus_shared={}]"'
                .format(command_string, self.num_gpus, self.num_gpus_shared)
                )
        command_string = '{} {}'.format(command_string, self.command)
        return command_string

    def submit_command(self):
        import subprocess
        submit_command = self.make_command_list()
        subprocess.call(submit_command)
