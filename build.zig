const std = @import("std");
const Build = std.Build;
const builtin = std.builtin;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const engine_module = b.dependency("engine", .{
        .target = target,
        .optimize = optimize,
    }).module("engine");
    const nterm_module = engine_module.import_table.get("nterm").?;

    const zmai_module = b.dependency("zmai", .{
        .target = target,
        .optimize = optimize,
    }).module("zmai");

    // Expose the library root
    _ = b.addModule("bot", .{
        .root_source_file = lazyPath(b, "src/root.zig"),
        .imports = &.{
            .{ .name = "engine", .module = engine_module },
            .{ .name = "zmai", .module = zmai_module },
        },
    });

    const install_NNs = b.addInstallDirectory(.{
        .source_dir = lazyPath(b, "NNs"),
        .install_dir = .bin,
        .install_subdir = "NNs",
    });

    buildExe(b, target, optimize, engine_module, nterm_module, zmai_module, install_NNs);
    buildTests(b, engine_module, zmai_module, install_NNs);
    buildBench(b, target, engine_module, zmai_module, install_NNs);
}

fn buildExe(
    b: *std.Build,
    target: Build.ResolvedTarget,
    optimize: builtin.OptimizeMode,
    engine_module: *Build.Module,
    nterm_module: *Build.Module,
    zmai_module: *Build.Module,
    install_NNs: *Build.Step.InstallDir,
) void {
    const train_exe = b.addExecutable(.{
        .name = "Budget Tetris Bot Training",
        .root_source_file = lazyPath(b, "src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    train_exe.root_module.addImport("engine", engine_module);
    train_exe.root_module.addImport("nterm", nterm_module);
    train_exe.root_module.addImport("zmai", zmai_module);

    b.installArtifact(train_exe);

    // Add run step
    const run_cmd = b.addRunArtifact(train_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    train_exe.step.dependOn(&install_NNs.step);
    run_step.dependOn(&run_cmd.step);
}

fn buildTests(
    b: *Build,
    engine_module: *Build.Module,
    zmai_module: *Build.Module,
    install_NNs: *Build.Step.InstallDir,
) void {
    const lib_tests = b.addTest(.{
        .root_source_file = lazyPath(b, "src/root.zig"),
    });
    lib_tests.root_module.addImport("engine", engine_module);
    lib_tests.root_module.addImport("zmai", zmai_module);

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&install_NNs.step);
    test_step.dependOn(&run_lib_tests.step);
}

fn buildBench(
    b: *Build,
    target: Build.ResolvedTarget,
    engine_module: *Build.Module,
    zmai_module: *Build.Module,
    install_NNs: *Build.Step.InstallDir,
) void {
    const bench_exe = b.addExecutable(.{
        .name = "Budget Tetris Bot Benchmarks",
        .root_source_file = lazyPath(b, "src/bench.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_exe.root_module.addImport("engine", engine_module);
    bench_exe.root_module.addImport("zmai", zmai_module);

    b.installArtifact(bench_exe);

    const bench_cmd = b.addRunArtifact(bench_exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        bench_cmd.addArgs(args);
    }
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&install_NNs.step);
    bench_step.dependOn(&bench_cmd.step);
}

fn lazyPath(b: *Build, path: []const u8) Build.LazyPath {
    return .{
        .src_path = .{
            .owner = b,
            .sub_path = path,
        },
    };
}
