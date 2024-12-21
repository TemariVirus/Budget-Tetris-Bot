const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;

const engine = @import("engine");
const AttackTable = engine.attack.AttackTable;
const BoardMask = engine.bit_masks.BoardMask;
// TODO: Replace with `anytype`
const GameState = engine.GameState(engine.bags.SevenBag);
const Facing = engine.pieces.Facing;
const PieceKind = engine.pieces.PieceKind;
const Rotation = engine.kicks.Rotation;

const root = @import("../root.zig");
const NN = root.neat.NN(8, 2);
const Placement = root.Placement;
const allPlacements = root.movegen.allPlacements;
const getFeaturesFull = root.neat.Bot.getFeaturesFull;

const Self = @This();

const BEAM_WIDTH = 1000;
const MAX_DEPTH = 12;

const DISCOUNT_FACTOR = 0.95;
const DISCOUNTS: [MAX_DEPTH]f32 = blk: {
    var discounts: [MAX_DEPTH]f32 = undefined;
    for (0..MAX_DEPTH) |i| {
        discounts[i] = std.math.pow(f32, DISCOUNT_FACTOR, i);
    }
    break :blk discounts;
};

network: NN,
think_nanos: u128,
attack_table: AttackTable,

current_depth: u32 = 0,
_current_depth: u32 = 0,
node_count: u64 = 0,
_node_count: u64 = 0,
memory_usage: u64 = 0,

// TODO: Store beams in a binary heap, sorting by score. Use hashmap to map beam to index in heap
// TODO: Have an iter function to find all beems with a certain score. If an identical beam is found, don't insert the current beam
const Beam = struct {
    placement: Placement,
    game: GameState,
    attack: f32,
    cleared: u32,
    intent: f32,
    score: f32,

    fn greaterThan(_: void, self: Beam, other: Beam) bool {
        return self.score > other.score;
    }
};

pub fn init(network: NN, think_seconds: f64, attack_table: AttackTable) Self {
    return .{
        .network = network,
        .think_nanos = @intFromFloat(think_seconds * std.time.ns_per_s),
        .attack_table = attack_table,
    };
}

pub fn findMoves(self: *Self, allocator: Allocator, game: GameState) !Placement {
    const start_time = std.time.nanoTimestamp();
    self._node_count = 0;

    var beams = try std.ArrayList(Beam).initCapacity(allocator, BEAM_WIDTH);
    defer beams.deinit();
    var new_beams = try std.ArrayList(Beam).initCapacity(allocator, BEAM_WIDTH);
    defer new_beams.deinit();

    // Depth 0
    const initial_out = self.network.predict(getFeaturesFull(
        game.playfield,
        self.network.inputs_used[0..5].*,
        0,
        0,
        0,
    ));
    try self.searchBeam(.{
        .placement = undefined,
        .game = game,
        .attack = 0,
        .cleared = 0,
        .intent = initial_out[1],
        .score = initial_out[0],
    }, 0, &new_beams);
    self.commitBeams(&beams, &new_beams);

    // Depth 1 and onwards
    outer: for (1..MAX_DEPTH) |depth| {
        self._current_depth = @intCast(depth);

        for (beams.items) |beam| {
            if (std.time.nanoTimestamp() - start_time > self.think_nanos) {
                break :outer;
            }

            try self.searchBeam(beam, depth, &new_beams);
        }
        self.commitBeams(&beams, &new_beams);
    } else {
        self._current_depth = MAX_DEPTH;
    }

    self.current_depth = self._current_depth;
    self.node_count = self._node_count;
    self.memory_usage = (beams.capacity + new_beams.capacity) * @sizeOf(Beam) / 1024;
    return beams.items[0].placement;
}

fn searchBeam(self: Self, beam: Beam, depth: usize, beams: *std.ArrayList(Beam)) !void {
    for (0..2) |i| {
        const use_hold = i == 1;
        var clone1 = beam.game;
        if (use_hold) {
            clone1.hold();
            if (clone1.current.kind == clone1.hold_kind) {
                continue;
            }
        }

        const placements = allPlacements(
            clone1.playfield,
            clone1.kicks,
            clone1.current.kind,
            @min(21, clone1.playfield.height() +| 4),
        );

        var iter = placements.iterator(clone1.current.kind);
        while (iter.next()) |placement| {
            var clone2 = clone1;
            clone2.current = placement.piece;
            clone2.pos = placement.pos;
            const clear_info = clone2.lockCurrent(4); // 4th kick needed for some T-Spin Triples
            const attack: f32 = @floatFromInt(self.attack_table.getAttack(
                clear_info,
                clone2.b2b,
                clone2.combo,
            ));

            const new_attack = DISCOUNTS[depth] * attack + beam.attack;
            const new_cleared = clear_info.cleared + beam.cleared;
            const features = getFeaturesFull(
                clone2.playfield,
                self.network.inputs_used[0..5].*,
                new_attack,
                new_cleared,
                beam.intent,
            );
            const outputs = self.network.predict(features);
            try beams.append(.{
                .placement = if (depth == 0) placement else beam.placement,
                .game = clone2,
                .attack = new_attack,
                .cleared = new_cleared,
                .intent = outputs[1],
                .score = outputs[0],
            });
        }
    }
}

fn commitBeams(self: *Self, beams: *std.ArrayList(Beam), new_beams: *std.ArrayList(Beam)) void {
    const beam_count = @min(beams.capacity, new_beams.items.len);
    beams.items.len = beam_count;
    @memcpy(beams.items[0..beam_count], new_beams.items[0..beam_count]);
    std.sort.pdq(Beam, beams.items, {}, Beam.greaterThan);

    self._node_count += new_beams.items.len;
    new_beams.clearRetainingCapacity();
}
