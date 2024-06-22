const std = @import("std");
const time = std.time;

const engine = @import("engine");
const SevenBag = engine.bags.SevenBag;
const GameState = engine.GameState(SevenBag);

const root = @import("root.zig");
const pc = root.pc;
const Bot = root.neat.Bot;

pub fn main() !void {
    try pcBenchmark(4);
    try nnBenchmark();
    getFeaturesBenchmark();
}

// Height: 4
// Mean: 54.967ms
// Max: 809.671ms
pub fn pcBenchmark(comptime height: u8) !void {
    const NN = root.neat.NN(5, 1);
    const RUN_COUNT = 100;

    std.debug.print(
        \\
        \\------------------
        \\   PC Benchmark
        \\------------------
        \\
    , .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const nn = try NN.load(allocator, "NNs/PC.json");
    defer nn.deinit(allocator);

    const start = time.nanoTimestamp();
    var max_time: u64 = 0;
    for (0..RUN_COUNT) |seed| {
        const gamestate = GameState.init(SevenBag.init(seed), engine.kicks.srsPlus);

        const solve_start = time.nanoTimestamp();
        const solution = try pc.findPc(allocator, gamestate, nn, height, height * 10 / 4);
        defer allocator.free(solution);

        const time_taken: u64 = @intCast(time.nanoTimestamp() - solve_start);
        max_time = @max(max_time, time_taken);
        std.mem.doNotOptimizeAway(solution);

        std.debug.print(
            "Seed: {:<2} | Time taken: {}\n",
            .{ seed, std.fmt.fmtDuration(time_taken) },
        );
    }
    const total_time: u64 = @intCast(time.nanoTimestamp() - start);

    std.debug.print("Mean: {}\n", .{std.fmt.fmtDuration(total_time / RUN_COUNT)});
    std.debug.print("Max: {}\n", .{std.fmt.fmtDuration(max_time)});
}

// Mean: 45ns
// Iters/s: 21772159
pub fn nnBenchmark() !void {
    const NN = root.neat.NN(8, 2);
    const RUN_COUNT = 300_000_000;

    std.debug.print(
        \\
        \\------------------
        \\   NN Benchmark
        \\------------------
        \\
    , .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const nn = try NN.load(allocator, "NNs/Qoshae.json");
    defer nn.deinit(allocator);

    const start = std.time.nanoTimestamp();
    for (0..RUN_COUNT) |_| {
        const out = nn.predict([_]f32{ 5.2, 1.0, 3.0, 9.0, 11.0, 5.0, 2.0, -0.97 });
        std.mem.doNotOptimizeAway(out);
    }
    const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);

    std.debug.print("Mean: {}\n", .{std.fmt.fmtDuration(time_taken / RUN_COUNT)});
    std.debug.print("Iters/s: {}\n", .{std.time.ns_per_s * RUN_COUNT / time_taken});
}

// Mean: 81ns
// Iters/s: 12278357
pub fn getFeaturesBenchmark() void {
    const RUN_COUNT = 200_000_000;

    std.debug.print(
        \\
        \\--------------------------------
        \\  Feature Extraction Benchmark
        \\--------------------------------
        \\
    , .{});

    // Randomly place 3 pieces
    var xor = std.Random.Xoroshiro128.init(0);
    const rand = xor.random();
    var game = GameState.init(SevenBag.init(xor.next()), engine.kicks.srsPlus);
    for (0..3) |_| {
        game.current.facing = rand.enumValue(engine.pieces.Facing);
        game.pos.x = rand.intRangeAtMost(i8, game.current.minX(), game.current.maxX());
        _ = game.dropToGround();
        _ = game.lockCurrent(-1);
    }

    const start = time.nanoTimestamp();
    for (0..RUN_COUNT) |_| {
        std.mem.doNotOptimizeAway(
            Bot.getFeatures(game.playfield, [_]bool{true} ** 5),
        );
    }
    const time_taken: u64 = @intCast(time.nanoTimestamp() - start);

    std.debug.print("Mean: {}\n", .{std.fmt.fmtDuration(time_taken / RUN_COUNT)});
    std.debug.print("Iters/s: {}\n", .{std.time.ns_per_s * RUN_COUNT / time_taken});
}
