const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const expect = std.testing.expect;

const engine = @import("engine");
const BoardMask = engine.bit_masks.BoardMask;
const GameState = engine.GameState(SevenBag);
const KickFn = engine.kicks.KickFn;
const PieceKind = engine.pieces.PieceKind;
const SevenBag = engine.bags.SevenBag;

const root = @import("root.zig");
const movegen = root.movegen;
const NN = root.neat.NN(5, 1);
const Placement = root.Placement;

const SearchNode = struct {
    rows: [23]u16,
    depth: u8,
};
const NodeSet = std.AutoHashMap(SearchNode, void);

pub const FindPcError = error{
    NoPcExists,
    NotEnoughPieces,
};

/// Finds a perfect clear with the least number of pieces possible for the given
/// game state, and returns the sequence of placements required to achieve it.
///
/// Returns an error if no perfect clear exists, or if the number of pieces needed
/// exceeds `max_pieces`.
pub fn findPc(
    allocator: Allocator,
    game: GameState,
    nn: NN,
    min_height: u6,
    max_pieces: u8,
) ![]Placement {
    const field_height = blk: {
        var i: usize = BoardMask.HEIGHT;
        while (i >= 1) : (i -= 1) {
            if (game.playfield.rows[i - 1] != BoardMask.EMPTY_ROW) {
                break;
            }
        }
        break :blk i;
    };
    const bits_set = blk: {
        var set: usize = 0;
        for (0..field_height) |i| {
            set += @popCount(game.playfield.rows[i] & ~BoardMask.EMPTY_ROW);
        }
        break :blk set;
    };
    const empty_cells = BoardMask.WIDTH * field_height - bits_set;

    // Assumes that all pieces have 4 cells and that the playfield is 10 cells wide.
    // Thus, an odd number of empty cells means that a perfect clear is impossible.
    if (empty_cells % 2 == 1) {
        return FindPcError.NoPcExists;
    }

    var pieces_needed = if (empty_cells % 4 == 2)
        // If the number of empty cells is not a multiple of 4, we need to fill
        // an extra so that it becomes a multiple of 4
        // 2 + 10 = 12 which is a multiple of 4
        (empty_cells + 10) / 4
    else
        empty_cells / 4;
    // Don't return an empty solution
    if (pieces_needed == 0) {
        pieces_needed = 5;
    }

    const pieces = try getPieces(allocator, game, max_pieces + 1);
    defer allocator.free(pieces);

    var cache = NodeSet.init(allocator);
    defer cache.deinit();

    // 20 is the lowest common multiple of the width of the playfield (10) and the
    // number of cells in a piece (4). 20 / 4 = 5 extra pieces for each bigger
    // perfect clear
    while (pieces_needed <= pieces.len) : (pieces_needed += 5) {
        const max_height = (4 * pieces_needed + bits_set) / BoardMask.WIDTH;
        if (max_height < min_height) {
            continue;
        }

        const placements = try allocator.alloc(Placement, pieces_needed);
        errdefer allocator.free(placements);

        // Pre-allocate a queue for each placement
        const queues = try allocator.alloc(movegen.MoveQueue, pieces_needed);
        for (0..queues.len) |i| {
            queues[i] = movegen.MoveQueue.init(allocator, {});
        }
        defer allocator.free(queues);
        defer for (queues) |queue| {
            queue.deinit();
        };

        cache.clearRetainingCapacity();
        if (findPcInner(
            game.playfield,
            pieces,
            queues,
            placements,
            game.kicks,
            &cache,
            nn,
            @intCast(max_height),
        )) {
            return placements;
        }

        allocator.free(placements);
    }

    return FindPcError.NotEnoughPieces;
}

/// Extracts `pieces_count` pieces from the game state, in the format [current, hold, next...].
pub fn getPieces(allocator: Allocator, game: GameState, pieces_count: usize) ![]PieceKind {
    if (pieces_count == 0) {
        return &.{};
    }

    var pieces = try allocator.alloc(PieceKind, pieces_count);
    pieces[0] = game.current.kind;
    if (pieces_count == 1) {
        return pieces;
    }

    const start: usize = if (game.hold_kind) |hold| blk: {
        pieces[1] = hold;
        break :blk 2;
    } else 1;

    for (game.next_pieces, start..) |piece, i| {
        if (i >= pieces.len) {
            break;
        }
        pieces[i] = piece;
    }

    // If next pieces are not enough, fill the rest from the bag
    var bag_copy = game.bag;
    for (@min(pieces.len, start + game.next_pieces.len)..pieces.len) |i| {
        pieces[i] = bag_copy.next();
    }

    return pieces;
}

fn findPcInner(
    playfield: BoardMask,
    pieces: []PieceKind,
    queues: []movegen.MoveQueue,
    placements: []Placement,
    kick_fn: *const KickFn,
    cache: *NodeSet,
    nn: NN,
    max_height: u6,
) bool {
    // Base case; check for perfect clear
    if (placements.len == 0) {
        return max_height == 0;
    }

    const node = SearchNode{
        .rows = playfield.rows[0..23].*,
        .depth = @intCast(placements.len - 1),
    };
    if ((cache.getOrPut(node) catch unreachable).found_existing) {
        return false;
    }

    // Add moves to queue
    queues[0].items.len = 0;
    const m1 = movegen.allPlacements(playfield, kick_fn, pieces[0], max_height);
    movegen.orderMoves(
        &queues[0],
        playfield,
        pieces[0],
        m1,
        max_height,
        isPcPossible,
        nn,
        orderScore,
    );
    // Check for unique hold
    if (pieces.len > 1 and pieces[0] != pieces[1]) {
        const m2 = movegen.allPlacements(playfield, kick_fn, pieces[1], max_height);
        movegen.orderMoves(
            &queues[0],
            playfield,
            pieces[1],
            m2,
            max_height,
            isPcPossible,
            nn,
            orderScore,
        );
    }

    var held_odd_times = false;
    while (queues[0].removeOrNull()) |move| {
        const placement = move.placement;
        // Hold if needed
        if (placement.piece.kind != pieces[0]) {
            std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
            held_odd_times = !held_odd_times;
        }
        assert(pieces[0] == placement.piece.kind);

        var board = playfield;
        board.place(placement.piece.mask(), placement.pos);
        const cleared = board.clearLines(placement.pos.y);

        const new_height = max_height - cleared;
        if (findPcInner(
            board,
            pieces[1..],
            queues[1..],
            placements[1..],
            kick_fn,
            cache,
            nn,
            new_height,
        )) {
            placements[0] = placement;
            return true;
        }
    }
    // Unhold if held an odd number of times so that pieces are in the same order
    if (held_odd_times) {
        std.mem.swap(PieceKind, &pieces[0], &pieces[1]);
    }

    return false;
}

fn isPcPossible(rows: []const u16) bool {
    var walls = ~BoardMask.EMPTY_ROW;
    for (rows) |row| {
        walls &= row | (row << 1);
    }
    walls &= walls ^ (walls >> 1); // Reduce consecutive walls to 1 wide walls

    while (walls != 0) {
        const old_walls = walls;
        walls &= walls - 1; // Clear lowest bit
        // A mask of all the bits before the removed wall
        const right_of_wall = (walls ^ old_walls) - 1;

        // Each "segment" separated by a wall must have a multiple of 4 empty cells,
        // as pieces can only be placed in one segment (each piece occupies 4 cells).
        var empty_count: u16 = 0;
        for (rows) |row| {
            // All of the other segments to the right are confirmed to have a
            // multiple of 4 empty cells, so it doesn't matter if we count them again.
            const segment = ~row & right_of_wall;
            empty_count += @popCount(segment);
        }
        if (empty_count % 4 != 0) {
            return false;
        }
    }

    return true;
}

fn orderScore(playfield: BoardMask, nn: NN) f32 {
    const features = root.neat.Bot.getFeatures(playfield, nn.inputs_used);
    return nn.predict(features)[0];
}

test "4-line PC" {
    const allocator = std.testing.allocator;

    var gamestate = GameState.init(SevenBag.init(0), engine.kicks.srsPlus);

    const nn = try NN.load(allocator, "NNs/PC.json");
    defer nn.deinit(allocator);

    const solution = try findPc(allocator, gamestate, nn, 0, 10);
    defer allocator.free(solution);
    try expect(solution.len == 10);

    for (solution[0 .. solution.len - 1]) |placement| {
        gamestate.current = placement.piece;
        gamestate.pos = placement.pos;
        try expect(!gamestate.lockCurrent(-1).pc);
    }

    gamestate.current = solution[solution.len - 1].piece;
    gamestate.pos = solution[solution.len - 1].pos;
    try expect(gamestate.lockCurrent(-1).pc);
}
