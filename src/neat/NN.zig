const std = @import("std");
const Allocator = std.mem.Allocator;

const NNInner = @import("zmai").genetic.neat.NN;

pub fn NN(comptime input_count: usize, comptime output_count: usize) type {
    return struct {
        const Self = @This();

        net: NNInner,
        inputs_used: [input_count]bool,
        comptime input_count: usize = input_count,
        comptime output_count: usize = output_count,

        pub fn load(allocator: Allocator, path: []const u8) !Self {
            var inputs_used: [input_count]bool = undefined;
            const nn = try NNInner.load(allocator, path, &inputs_used);
            return .{
                .net = nn,
                .inputs_used = inputs_used,
            };
        }

        pub fn deinit(self: Self, allocator: Allocator) void {
            self.net.deinit(allocator);
        }

        pub fn predict(self: Self, input: [input_count]f32) [output_count]f32 {
            var output: [output_count]f32 = undefined;
            self.net.predict(&input, &output);
            return output;
        }
    };
}
