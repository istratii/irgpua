#pragma once

#define ALIGN32(N) ((N + 31) & ~31)
#define ALIGN128(N) ((N + 127) & ~127)