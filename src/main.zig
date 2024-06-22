const std = @import("std");
const Allocator = std.mem.Allocator;
const SolutionList = std.ArrayList([]Placement);
const time = std.time;

const engine = @import("engine");
const Player = engine.Player(SevenBag);
const GameState = Player.GameState;
const kicks = engine.kicks;
const PeriodicTrigger = engine.PeriodicTrigger;
const SevenBag = engine.bags.SevenBag;

const nterm = @import("nterm");
const View = nterm.View;

const root = @import("root.zig");
const Bot = root.neat.Bot;
const NN = root.neat.NN(8, 2);
const pc = root.pc;
const PcNn = root.neat.NN(5, 1);
const Placement = root.Placement;

/// Also used as the number of placements/s
const FRAMERATE = 20;
const FPS_TIMING_WINDOW = FRAMERATE * 2;
/// The maximum number of perfect clears to calculate in advance.
const MAX_PC_QUEUE = 16;

// pub fn main() !void {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     // Add 2 to create a 1-wide empty boarder on the left and right.
//     try nterm.init(
//         allocator,
//         std.io.getStdOut(),
//         FPS_TIMING_WINDOW,
//         Player.DISPLAY_W + 2,
//         Player.DISPLAY_H + 3,
//     );
//     defer nterm.deinit();

//     const settings = engine.GameSettings{
//         .g = 0,
//         .target_mode = .none,
//     };
//     const player_view = View{
//         .left = 1,
//         .top = 0,
//         .width = Player.DISPLAY_W,
//         .height = Player.DISPLAY_H,
//     };
//     var player = Player.init(
//         "Bot",
//         SevenBag.init(0),
//         kicks.srsPlus,
//         settings,
//         player_view,
//         playSfxDummy,
//     );

//     const bot_stats_view = View{
//         .left = 1,
//         .top = Player.DISPLAY_H,
//         .width = Player.DISPLAY_W + 1,
//         .height = 3,
//     };

//     const nn = try NN.load(allocator, "NNs/Qoshae.json");
//     defer nn.deinit(allocator);

//     var bot = Bot.init(nn, 0.5, player.settings.attack_table);

//     var t = time.nanoTimestamp();
//     while (true) {
//         const placement = bot.findMoves(player.state);
//         if (placement.piece.kind != player.state.current.kind) {
//             player.hold();
//         }
//         player.state.pos = placement.pos;
//         player.state.current = placement.piece;
//         player.hardDrop(0, &.{});

//         bot_stats_view.printAt(0, 0, .white, .black, "Nodes: {d}", .{bot.node_count});
//         bot_stats_view.printAt(0, 1, .white, .black, "Depth: {d}", .{bot.current_depth});
//         bot_stats_view.printAt(0, 2, .white, .black, "Tresh: {d}", .{bot.move_tresh});

//         const dt: u64 = @intCast(time.nanoTimestamp() - t);
//         player.tick(dt, 0, &.{});
//         t += dt;

//         player.draw();
//         nterm.render() catch |err| {
//             if (err == error.NotInitialized) {
//                 return;
//             }
//             return err;
//         };
//     }
// }

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Add 2 to create a 1-wide empty boarder on the left and right.
    try nterm.init(
        allocator,
        std.io.getStdOut(),
        FPS_TIMING_WINDOW,
        Player.DISPLAY_W + 2,
        Player.DISPLAY_H,
    );
    defer nterm.deinit();

    const settings = engine.GameSettings{
        .g = 0,
        .target_mode = .none,
    };
    const player_view = View{
        .left = 1,
        .top = 0,
        .width = Player.DISPLAY_W,
        .height = Player.DISPLAY_H,
    };
    var player = Player.init(
        "PC Solver",
        SevenBag.init(0),
        kicks.srsPlus,
        settings,
        player_view,
        playSfxDummy,
    );

    var placement_i: usize = 0;
    var pc_queue = SolutionList.init(allocator);
    defer pc_queue.deinit();

    const pc_thread = try std.Thread.spawn(.{
        .allocator = allocator,
    }, pcThread, .{ allocator, player.state, &pc_queue });
    defer pc_thread.join();

    const fps_view = View{
        .left = 1,
        .top = 0,
        .width = 15,
        .height = 1,
    };

    var render_timer = PeriodicTrigger.init(time.ns_per_s / FRAMERATE);
    while (true) {
        if (render_timer.trigger()) |dt| {
            fps_view.printAt(0, 0, .white, .black, "{d:.2}FPS", .{nterm.fps()});

            placePcPiece(allocator, &player, &pc_queue, &placement_i);
            player.tick(dt, 0, &.{});
            player.draw();
            nterm.render() catch |err| {
                // Trying to render after the terminal has been closed results
                // in an error, in which case stop the program gracefully.
                if (err == error.NotInitialized) {
                    return;
                }
                return err;
            };
        } else {
            time.sleep(1 * time.ns_per_ms);
        }
    }
}

fn placePcPiece(
    allocator: Allocator,
    game: *Player,
    queue: *SolutionList,
    placement_i: *usize,
) void {
    if (queue.items.len == 0) {
        return;
    }
    const placements = queue.items[0];

    const placement = placements[placement_i.*];
    if (placement.piece.kind != game.state.current.kind) {
        game.hold();
    }
    game.state.pos = placement.pos;
    game.state.current = placement.piece;
    game.hardDrop(0, &.{});
    placement_i.* += 1;

    // Start next perfect clear
    if (placement_i.* == placements.len) {
        allocator.free(queue.orderedRemove(0));
        placement_i.* = 0;
    }
}

fn pcThread(allocator: Allocator, state: GameState, queue: *SolutionList) !void {
    var game = state;

    const nn = try PcNn.load(allocator, "NNs/PC.json");
    defer nn.deinit(allocator);

    while (true) {
        while (queue.items.len >= MAX_PC_QUEUE) {
            time.sleep(time.ns_per_ms);
        }

        // A 2- or 4-line PC is not always possible. 15 placements is enough
        // for a 6-line PC.
        const solution = try pc.findPc(allocator, game, nn, 0, 15);
        for (solution) |placement| {
            if (game.current.kind != placement.piece.kind) {
                game.hold();
            }
            game.current = placement.piece;
            game.pos = placement.pos;
            _ = game.lockCurrent(-1);
            game.nextPiece();
        }

        try queue.append(solution);
    }
}

/// Dummy function to satisfy the Player struct.
fn playSfxDummy(_: engine.player.Sfx) void {}
