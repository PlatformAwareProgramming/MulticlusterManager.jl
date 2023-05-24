# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# This file is a part of Julia. License is MIT: https://julialang.org/license

# Built-in SSH and Local Managers

# Explicit mpi programs
# [..., ("host name", "mpiprogram_manager.jl", n1, "mpiprogram_w1.jl",..., nk, "mpiprogram_slave_w2.jl"), ...]

# Using config file (it is assumed there is a manager file called configfilename.jl)
# [..., ("host name", "configfilename"), ...]


struct MulticlusterSSHManager <: Distributed.ClusterManager
    machines::Dict
    manager_program::String

    function MulticlusterSSHManager(machines)
        # machines => array of machine elements
        # machine => address or (address, cnt)
        # address => string of form `[user@]host[:port] bind_addr[:bind_port]`
        # cnt => :auto or number
        # :auto launches NUM_CORES number of workers at address
        # number launches the specified number of workers at address
        mhist = Dict()

        for m in machines
            (host, manager_program, worker_programs...) = m
            mhist[host] = (manager_program, worker_programs)
        end
        new(mhist)
    end
end

Distributed.default_addprocs_params(::MulticlusterSSHManager) =
    merge(Distributed.default_addprocs_params(),
          Dict{Symbol,Any}(
              :hostfile       => "/home/heron/hostfile",
              :ssh            => "ssh",
              :sshflags       => ``,
              :scpflags       => ``,
              :shell          => :posix,
              :cmdline_cookie => false,
              :env            => [],
              :tunnel         => false,
              :multiplex      => false,
              :which_mpi      => "MPICH",
              :max_parallel   => 10))


# MulticlusterSSHManager

# start and connect to processes via SSH, optionally through an SSH tunnel.
# the tunnel is only used from the head (process 1); the nodes are assumed
# to be mutually reachable without a tunnel, as is often the case in a cluster.
# Default value of kw arg max_parallel is the default value of MaxStartups in sshd_config
# A machine is either a <hostname> or a tuple of (<hostname>, count)
"""
    addprocs(machines; tunnel=false, sshflags=\`\`, max_parallel=10, kwargs...) -> List of process identifiers

Add worker processes on remote machines via SSH. Configuration is done with keyword
arguments (see below). In particular, the `exename` keyword can be used to specify
the path to the `julia` binary on the remote machine(s).

`machines` is a vector of "machine specifications" which are given as strings of
the form `[user@]host[:port] [bind_addr[:port]]`. `user` defaults to current user and `port`
to the standard SSH port. If `[bind_addr[:port]]` is specified, other workers will connect
to this worker at the specified `bind_addr` and `port`.

It is possible to launch multiple processes on a remote host by using a tuple in the
`machines` vector or the form `(machine_spec, count)`, where `count` is the number of
workers to be launched on the specified host. Passing `:auto` as the worker count will
launch as many workers as the number of CPU threads on the remote host.

**Examples**:
```julia
addprocs([
    "remote1",               # one worker on 'remote1' logging in with the current username
    "user@remote2",          # one worker on 'remote2' logging in with the 'user' username
    "user@remote3:2222",     # specifying SSH port to '2222' for 'remote3'
    ("user@remote4", 4),     # launch 4 workers on 'remote4'
    ("user@remote5", :auto), # launch as many workers as CPU threads on 'remote5'
])
```

**Keyword arguments**:

* `tunnel`: if `true` then SSH tunneling will be used to connect to the worker from the
  master process. Default is `false`.

* `multiplex`: if `true` then SSH multiplexing is used for SSH tunneling. Default is `false`.

* `ssh`: the name or path of the SSH client executable used to start the workers.
  Default is `"ssh"`.

* `sshflags`: specifies additional ssh options, e.g. ``` sshflags=\`-i /home/foo/bar.pem\` ```

* `max_parallel`: specifies the maximum number of workers connected to in parallel at a
  host. Defaults to 10.

* `shell`: specifies the type of shell to which ssh connects on the workers.

    + `shell=:posix`: a POSIX-compatible Unix/Linux shell
      (sh, ksh, bash, dash, zsh, etc.). The default.

    + `shell=:csh`: a Unix C shell (csh, tcsh).

    + `shell=:wincmd`: Microsoft Windows `cmd.exe`.

* `dir`: specifies the working directory on the workers. Defaults to the host's current
  directory (as found by `pwd()`)

* `enable_threaded_blas`: if `true` then  BLAS will run on multiple threads in added
  processes. Default is `false`.

* `exename`: name of the `julia` executable. Defaults to `"\$(Sys.BINDIR)/julia"` or
  `"\$(Sys.BINDIR)/julia-debug"` as the case may be. It is recommended that a common Julia
  version is used on all remote machines because serialization and code distribution might
  fail otherwise.

* `exeflags`: additional flags passed to the worker processes.

* `topology`: Specifies how the workers connect to each other. Sending a message between
  unconnected workers results in an error.

    + `topology=:all_to_all`: All processes are connected to each other. The default.

    + `topology=:master_worker`: Only the driver process, i.e. `pid` 1 connects to the
      workers. The workers do not connect to each other.

    + `topology=:custom`: The `launch` method of the cluster manager specifies the
      connection topology via fields `ident` and `connect_idents` in `WorkerConfig`.
      A worker with a cluster manager identity `ident` will connect to all workers specified
      in `connect_idents`.

* `lazy`: Applicable only with `topology=:all_to_all`. If `true`, worker-worker connections
  are setup lazily, i.e. they are setup at the first instance of a remote call between
  workers. Default is true.

* `env`: provide an array of string pairs such as
  `env=["JULIA_DEPOT_PATH"=>"/depot"]` to request that environment variables
  are set on the remote machine. By default only the environment variable
  `JULIA_WORKER_TIMEOUT` is passed automatically from the local to the remote
  environment.

* `cmdline_cookie`: pass the authentication cookie via the `--worker` commandline
   option. The (more secure) default behaviour of passing the cookie via ssh stdio
   may hang with Windows workers that use older (pre-ConPTY) Julia or Windows versions,
   in which case `cmdline_cookie=true` offers a work-around.

!!! compat "Julia 1.6"
    The keyword arguments `ssh`, `shell`, `env` and `cmdline_cookie`
    were added in Julia 1.6.

Environment variables:

If the master process fails to establish a connection with a newly launched worker within
60.0 seconds, the worker treats it as a fatal situation and terminates.
This timeout can be controlled via environment variable `JULIA_WORKER_TIMEOUT`.
The value of `JULIA_WORKER_TIMEOUT` on the master process specifies the number of seconds a
newly launched worker waits for connection establishment.
"""
function addprocs_mpi(machines::AbstractVector; kwargs...)
    manager = MulticlusterSSHManager(machines)
    Distributed.check_addprocs_args(manager, kwargs)
    wids = Distributed.addprocs(manager; kwargs...)
    for wid in wids
        connectMPIWorkers(wid)
    end
end

function connectMPIWorkers(id)
     wconfig = Distributed.worker_from_id(id).config
     try
        Distributed.remotecall_fetch(Core.eval, id, Main, :(using Hash))
        Distributed.remotecall_fetch(Core.eval, id, Main, :(Hash.setRank($id)))
        Distributed.remotecall_fetch(Core.eval, id, Main, :(Hash.mpibcast_rank()))
        Distributed.remotecall_fetch(Core.eval, id, Main, :(include($(wconfig.userdata))))
     catch err
        #if isa(err, Distributed.RemoteException)
            @error "remote exception captured: $err"
        #else
        #    rethrow()
        #end
     end
end

#=
function Distributed.ssh_tunnel(user, host, bind_addr, port, sshflags, multiplex)
    port = Int(port)
    cnt = ntries = 100

    # the connection is forwarded to `port` on the remote server over the local port `localport`
    while cnt > 0
        localport = Distributed.next_tunnel_port()
        if multiplex
            # It assumes that an ssh multiplexing session has been already started by the remote worker.
            cmd = `ssh $sshflags -O forward -L $localport:$bind_addr:$port $user@$host`
        else
            # if we cannot do port forwarding, fail immediately
            # the -f option backgrounds the ssh session
            # `sleep 60` command specifies that an allotted time of 60 seconds is allowed to start the
            # remote julia process and establish the network connections specified by the process topology.
            # If no connections are made within 60 seconds, ssh will exit and an error will be printed on the
            # process that launched the remote process.
            ssh = `ssh -T -a -x -o ExitOnForwardFailure=yes`
            cmd = detach(`$ssh -f $sshflags $user@$host -L $localport:$bind_addr:$port sleep 60`)
        end
        @info "SSH TUNNEL: $cmd"
        if success(cmd)
            return localport
        end
        cnt -= 1
    end

    throw(ErrorException(
        string("unable to create SSH tunnel after ", ntries, " tries. No free port?")))
end


function Distributed.connect(manager::ClusterManager, pid::Int, config::WorkerConfig)
  #=  if config.connect_at !== nothing
        # this is a worker-to-worker setup call.
        return connect_w2w(pid, config)
    end =#

    # master connecting to workers
    if config.io === nothing && config.host !== nothing
        @info "CONNECT 1"
        pubhost = notnothing(config.host)
        port = notnothing(config.port)
        bind_addr = something(config.bind_addr, pubhost)
        @info "CONNECT 1 $pubhost   $port   $bind_addr"
    else
        @info "CONNECT 2 -- $(config.connect_at) -- config.host=$(config.host) -- config.port=$(config.port)"
        (bind_addr, port::Int) = config.connect_at !== nothing ? config.connect_at[6] : Distributed.read_worker_host_port(config.io)
        if (config.connect_at !== nothing)
            config.host = config.connect_at[1]
            config.sshflags = config.connect_at[2]
            config.tunnel = config.connect_at[3]
            config.multiplex = config.connect_at[4]
            config.max_parallel = config.connect_at[5]
        end
        pubhost = something(config.host, bind_addr)
        config.host = pubhost
        config.port = port
        @info "CONNECT 2 $pubhost   $port   $bind_addr"
    end

    tunnel = #=something(config.tunnel, false) =# true

    @info "CONNECT tunnel=$tunnel"

    s = split(pubhost,'@')
    user = ""
    if length(s) > 1
        user = s[1]
        pubhost = s[2]
    else
        if haskey(ENV, "USER")
            user = ENV["USER"]
        elseif tunnel
            error("USER must be specified either in the environment ",
                  "or as part of the hostname when tunnel option is used")
        end
    end

    if tunnel
        if !haskey(Distributed.tunnel_hosts_map, pubhost)
            Distributed.tunnel_hosts_map[pubhost] = Base.Semaphore(something(config.max_parallel, typemax(Int)))
        end
        sem = Distributed.tunnel_hosts_map[pubhost]

        sshflags = Base.notnothing(config.sshflags)
        multiplex = something(config.multiplex, false)
        Base.acquire(sem)
        try
            @info "CONNECT 3 - BEGIN"
            (s, bind_addr, forward) = Distributed.connect_to_worker_with_tunnel(pubhost, bind_addr, port, user, sshflags, multiplex)
            config.forward = forward
            @info "CONNECT 3 - END -- forward=$forward"
        finally
            Base.release(sem)
        end
    else
        (s, bind_addr) = Distributed.connect_to_worker(bind_addr, port)
    end

    config.bind_addr = bind_addr

    # write out a subset of the connect_at required for further worker-worker connection setups
    config.connect_at = (config.host, config.sshflags, config.tunnel, config.multiplex, config.max_parallel, (bind_addr, port))

    if config.io !== nothing
        let pid = pid
            Distributed.redirect_worker_output(pid, Base.notnothing(config.io))
        end
    end

    (s, s)
end 
=#

function Distributed.launch(manager::MulticlusterSSHManager, params::Dict, launched::Array, launch_ntfy::Condition)
    # Launch one worker on each unique host in parallel. Additional workers are launched later.
    # Wait for all launches to complete.

    @sync for (i, (machine, (manager_program, worker_programs))) in enumerate(manager.machines)
        let machine=machine, manager_program=manager_program, worker_programs=worker_programs
             @async try
                Distributed.launch_on_machine(manager, $machine, $manager_program, $worker_programs, params, launched, launch_ntfy)
            catch e
                print(stderr, "exception launching on machine $(machine) : $(e)\n")
            end
        end
    end
    notify(launch_ntfy)
end


Base.show(io::IO, manager::MulticlusterSSHManager) = print(io, "MulticlusterSSHManager(machines=", manager.machines, ")")


function parse_machine(machine::AbstractString)
    hoststr = ""
    portnum = nothing

    if machine[begin] == '['  # ipv6 bracket notation (RFC 2732)
        ipv6_end = findlast(']', machine)
        if ipv6_end === nothing
            throw(ArgumentError("invalid machine definition format string: invalid port format \"$machine\""))
        end
        hoststr = machine[begin+1 : prevind(machine,ipv6_end)]
        machine_def = split(machine[ipv6_end : end] , ':')
    else    # ipv4
        machine_def = split(machine, ':')
        hoststr = machine_def[1]
    end

    if length(machine_def) > 2
        throw(ArgumentError("invalid machine definition format string: invalid port format \"$machine_def\""))
    end

    if length(machine_def) == 2
        portstr = machine_def[2]

        portnum = tryparse(Int, portstr)
        if portnum === nothing
            msg = "invalid machine definition format string: invalid port format \"$machine_def\""
            throw(ArgumentError(msg))
        end

        if portnum < 1 || portnum > 65535
            msg = "invalid machine definition format string: invalid port number \"$machine_def\""
            throw(ArgumentError(msg))
        end
    end
    (hoststr, portnum)
end

function Distributed.launch_on_machine(manager::MulticlusterSSHManager, machine::AbstractString, manager_program, worker_programs, params::Dict, launched::Array, launch_ntfy::Condition)
    shell = params[:shell]
    ssh = params[:ssh]
    dir = params[:dir]
    exename = params[:exename]
    exeflags = params[:exeflags]
    tunnel = params[:tunnel]
    multiplex = params[:multiplex]
    cmdline_cookie = params[:cmdline_cookie]
    which_mpi = params[:which_mpi]

    @info "hosfile ???"
    hostfile = haskey(params, :hostfile) ? params[:hostfile] : "~/hostfile"
    @info "hosfile is $hostfile"
    env = Dict{String,String}(params[:env])

    # machine could be of the format [user@]host[:port] bind_addr[:bind_port]
    # machine format string is split on whitespace
    machine_bind = split(machine)
    
    if isempty(machine_bind)
        throw(ArgumentError("invalid machine definition format string: \"$machine\$"))
    end

    if length(machine_bind) > 1
        exeflags = `--bind-to $(machine_bind[2]) $exeflags`
    end

    if cmdline_cookie
        exeflags = `$exeflags --worker=$(cluster_cookie())`
    else
        exeflags = `$exeflags --worker`
    end

    worker_programs = [i for i in worker_programs]
    @info worker_programs

    mpiexec_env = if which_mpi == "OpenMPI"
                     `-x UCX_WARN_UNUSED_ENV_VARS=n`
                  else
                     `-env UCX_WARN_UNUSED_ENV_VARS n`
                  end

    other_processes =  ``
    while !isempty(worker_programs)
        msz = popfirst!(worker_programs)
        flr = popfirst!(worker_programs)
        other_processes = `$other_processes : -np $msz $mpiexec_env $(params[:exename]) $flr mpi`
    end

    host, portnum = parse_machine(machine_bind[1])
    portopt = portnum === nothing ? `` : `-p $portnum`
    sshflags = `$(params[:sshflags]) $portopt`

    @info sshflags
    # id = isnothing(identity_file) ? `` : `-i $identity_file`

    if tunnel
        # First it checks if ssh multiplexing has been already enabled and the master process is running.
        # If it's already running, later ssh sessions also use the same ssh multiplexing session even if
        # `multiplex` is not explicitly specified; otherwise the tunneling session launched later won't
        # go to background and hang. This is because of OpenSSH implementation.
        ccc = `$ssh $sshflags -O check $host`
        @info ccc
        if success(ccc)
            multiplex = true
        elseif multiplex
            # automatically create an SSH multiplexing session at the next SSH connection
            controlpath = "~/.ssh/julia-%r@%h:%p"
            sshflags = `$sshflags -o ControlMaster=auto -o ControlPath=$controlpath -o ControlPersist=no`
        end
    end

    # Build up the ssh command

    # pass on some environment variables by default
    for var in ["JULIA_WORKER_TIMEOUT"]
        if !haskey(env, var) && haskey(ENV, var)
            env[var] = ENV[var]
        end
    end

  
    # Julia process with passed in command line flag arguments
    if shell === :posix
        # ssh connects to a POSIX shell

        cmds = "exec mpiexec -hostfile $(Base.shell_escape_posixly(hostfile)) --map-by node $(Base.shell_escape_posixly(mpiexec_env)) -np 1 $(Base.shell_escape_posixly(exename)) $(Base.shell_escape_posixly(exeflags)) $(Base.shell_escape_posixly(other_processes))"
        # set environment variables
        for (var, val) in env
            occursin(r"^[a-zA-Z_][a-zA-Z_0-9]*\z", var) ||
                throw(ArgumentError("invalid env key $var"))
            cmds = "export $(var)=$(Base.shell_escape_posixly(val))\n$cmds"
        end
        cmds = "module load mpi/OpenMPI\n$cmds"
       # change working directory
        cmds = "cd -- $(Base.shell_escape_posixly(dir))\n$cmds"

        # shell login (-l) with string command (-c) to launch julia process
        remotecmd = Base.shell_escape_posixly(`sh -l -c $cmds`)

    elseif shell === :csh
        # ssh connects to (t)csh

        remotecmd = "exec mpiexec -hostfile $(shell_escape_csh(hostfile)) --map-by node -np 1 $(shell_escape_csh(exename)) $(shell_escape_csh(exeflags)) $(Base.shell_escape_posixly(other_processes))"

        # set environment variables
        for (var, val) in env
            occursin(r"^[a-zA-Z_][a-zA-Z_0-9]*\z", var) ||
                throw(ArgumentError("invalid env key $var"))
            remotecmd = "setenv $(var) $(shell_escape_csh(val))\n$remotecmd"
        end
        # change working directory
        if dir !== nothing && dir != ""
            remotecmd = "cd $(shell_escape_csh(dir))\n$remotecmd"
        end

    elseif shell === :wincmd
        # ssh connects to Windows cmd.exe

        any(c -> c == '"', exename) && throw(ArgumentError("invalid exename"))

        remotecmd = shell_escape_wincmd(escape_microsoft_c_args(exename, exeflags..., other_processes...))
        # change working directory
        if dir !== nothing && dir != ""
            any(c -> c == '"', dir) && throw(ArgumentError("invalid dir"))
            remotecmd = "pushd \"$(dir)\" && $remotecmd"
        end
        # set environment variables
        for (var, val) in env
            occursin(r"^[a-zA-Z0-9_()[\]{}\$\\/#',;\.@!?*+-]+\z", var) || throw(ArgumentError("invalid env key $var"))
            remotecmd = "set $(var)=$(shell_escape_wincmd(val))&& $remotecmd"
        end

    else
        throw(ArgumentError("invalid shell"))
    end

    # remote launch with ssh with given ssh flags / host / port information
    # -T → disable pseudo-terminal allocation
    # -a → disable forwarding of auth agent connection
    # -x → disable X11 forwarding
    # -o ClearAllForwardings → option if forwarding connections and
    #                          forwarded connections are causing collisions
    @info sshflags
    cmd = `$ssh -T -a -x -o ClearAllForwardings=yes $sshflags $host $remotecmd`

    @info cmd

    # launch the remote Julia process

    # detach launches the command in a new process group, allowing it to outlive
    # the initial julia process (Ctrl-C and teardown methods are handled through messages)
    # for the launched processes.
    io = open(detach(cmd), "r+")
    cmdline_cookie || Distributed.write_cookie(io)

    wconfig = WorkerConfig()
    wconfig.io = io.out
    wconfig.host = host
    wconfig.tunnel = tunnel
    wconfig.multiplex = multiplex
    wconfig.sshflags = sshflags
    wconfig.exeflags = exeflags
    wconfig.exename = exename
    wconfig.count = 1
    wconfig.max_parallel = params[:max_parallel]
    wconfig.enable_threaded_blas = params[:enable_threaded_blas]
    wconfig.userdata = manager_program

    push!(launched, wconfig)

    notify(launch_ntfy)
end


function Distributed.manage(manager::MulticlusterSSHManager, id::Integer, config::WorkerConfig, op::Symbol)
    id = Int(id)
    if op === :interrupt
        ospid = config.ospid
        if ospid !== nothing
            host = Base.notnothing(config.host)
            sshflags = Base.notnothing(config.sshflags)
            if !success(`ssh -T -a -x -o ClearAllForwardings=yes -n $sshflags $host "kill -2 $ospid"`)
                @error "Error sending a Ctrl-C to julia worker $id on $host"
            end
        else
            # This state can happen immediately after an addprocs
            @error "Worker $id cannot be presently interrupted."
        end
    end
end



#=

    fromconfig: sent by the manager process

=#
function addprocs(new_addresses::Vector{Tuple{String,Int}}, fromconfig)
    
    launched_q = Int[]  
    @sync begin
        for address in new_addresses
            (bind_addr, port) = address

            wconfig = WorkerConfig()
            for x in [:host, :tunnel, :multiplex, :sshflags, :exeflags, :exename, :enable_threaded_blas]
                Base.setproperty!(wconfig, x, Base.getproperty(fromconfig, x))
            end
            wconfig.bind_addr = bind_addr
            wconfig.port = port

            let wconfig=wconfig
                @async begin
                    pid = create_worker(manager, wconfig)
                    #remote_do(redirect_output_from_additional_worker, frompid, pid, port)
                    push!(launched_q, pid)
                end
            end
        end
    end
    sort!(launched_q)
end