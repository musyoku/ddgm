import sys, time, math

def get_progress_bar(current_step, total_steps, num_segments=30):
	str = "["
	base = total_steps // num_segments
	for seg in xrange(num_segments):
		if base * seg < current_step:
			str += "="
		else:
			str += "."
	str += "]"
	return str

def get_args(args):
	str = ""
	for key, value in args.iteritems():
		str += " - {}: {}".format(key, value)
	return str

def show_progress(current_step, total_steps, elapsed_time, args):
	digits = int(math.log10(total_steps)) + 1
	progress_bar = get_progress_bar(current_step, total_steps)
	prefix = "{0:>{1}}/{2} {3}".format(current_step, digits, total_steps, progress_bar)
	args = get_args(args)
	sys.stdout.write("\r{} - {}m{}".format(prefix, elapsed_time, args))
	sys.stdout.flush()