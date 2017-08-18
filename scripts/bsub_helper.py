"""
Gregory Way 2017
Tybalt - Variational Autoencoder on Pan Cancer Gene Expression
scripts/bsub_helper.py

Usage: Import only

    from bsub_helper import bsub_help
    b = bsub_help(command)

    b.make_command_list()    # To make a python list of commands
    b.make_command_string()  # To make a python string of commands
    b.submit_command()       # Directly submit bsub job to pmacs
"""


class bsub_help():
    def __init__(self, command, queue='gpu', num_gpus=2, num_gpus_shared=0,
                 walltime='0:10', error_file='std_err.txt',
                 output_file='std_out.txt'):
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
                '{} -R "select[ngpus>{}] rusage [ngpus_shared={}]"'
                .format(command_string, self.num_gpus, self.num_gpus_shared)
                )
        command_string = '{} {}'.format(command_string, ' '.join(self.command))
        return command_string

    def submit_command(self):
        import subprocess
        submit_command = self.make_command_string()
        subprocess.call(submit_command, shell=True)
