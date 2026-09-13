#pragma once

// 统一仪器日志模块(私有 fork,B4 前置)。
//
// 动机(B3 教训):散落的 fprintf(stderr,...) 逐条直写会被重定向管道背压卡住
// (实测单次可达 ~2.5ms),污染计时;且无法整体关闭,仪器税常驻(~0.5ms/cycle)。
//
// 特性:
//   1. 全局开关:  env LLAMA_TRACE=1(默认关。关闭时每次调用 = 2 个 relaxed load
//      + 一个可预测分支,热路径近零开销)
//   2. tag 选择:  env LLAMA_TRACE_TAGS=P0T,P0G(逗号分隔,精确匹配;缺省 = 全部)
//   3. 缓冲批量:  热路径只写内存缓冲;安全点调 tracelog::flush() 一次性 fwrite
//      (兜底:缓冲 ≥ kFlushCap 时就地 flush)
//   4. 重复节流:  同一调用点(key = fmt 字面量地址)内容与上一条完全相同 → 只计数;
//      内容变化时补打 "[dup xN]";聚合计数器类日志(每窗口内容必变)不受影响
//
// 用法:
//      TRACELOG("P0T", "[P0T] avg=%.2fms\n", v);   // 函数体内任意位置
//      tracelog::flush();                           // 周期性安全点(如 cycle 边界)
//
// 线程模型:内部一把互斥锁保护缓冲与节流表;调用点集中在 server 主线程与
// 少量 GPU 回调,争用可忽略。 flushed 数据写 fd 2(stderr),与现有日志管道一致。

#if defined(__cplusplus)

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tracelog {

inline constexpr size_t kLineMax  = 1024;        // 单行上限(截断)
inline constexpr size_t kFlushCap = 512 * 1024;  // 缓冲兜底上限

struct site_t {
    uint64_t   dups = 0;   // 连续重复条数
    std::string last;      // 上一条内容(整行)
};

struct state_t {
    std::atomic<bool>     on{false};
    std::atomic<uint64_t> mask{0};
    std::atomic<bool>     inited{false};

    std::mutex mtx;                                // 保护 buf/sites/tags
    std::string buf;
    std::vector<std::string> tag_names;            // id -> 名(仅诊断用)
    std::string tags_csv;                          // ",tag1,tag2," 形式;空 = 全部
    bool tags_all = true;

    std::unordered_map<uint64_t, site_t> sites;    // key = fmt 指针

    ~state_t() {
        // 进程退出兜底:先结算未上报的 dup,再 flush(静态析构顺序风险可接受:仅读自身成员)
        for (auto & kv : sites) {
            if (kv.second.dups > 0) {
                buf += "[trace] [dup x" + std::to_string(kv.second.dups) + " (pending)]\n";
                kv.second.dups = 0;
            }
        }
        if (!buf.empty()) {
            fwrite(buf.data(), 1, buf.size(), stderr);
            buf.clear();
        }
    }
};

inline state_t & st() { static state_t s; return s; }

inline void init_once() {
    if (st().inited.load(std::memory_order_relaxed)) {
        return;
    }
    // 首调竞态无害:重复解析结果一致,mask 由 reg() 在锁内按名重算
    const char * on_env   = getenv("LLAMA_TRACE");
    const char * tags_env = getenv("LLAMA_TRACE_TAGS");

    if (on_env != nullptr && on_env[0] != '\0' && strcmp(on_env, "0") != 0) {
        st().on.store(true, std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> lk(st().mtx);
    if (tags_env == nullptr || tags_env[0] == '\0') {
        st().tags_all = true;
        st().tags_csv.clear();
    } else {
        st().tags_all = false;
        st().tags_csv = ",";
        for (const char * p = tags_env; *p != '\0'; ++p) {
            st().tags_csv.push_back(*p == ',' ? ',' : *p);
        }
        if (st().tags_csv.back() != ',') {
            st().tags_csv.push_back(',');
        }
    }
    st().inited.store(true, std::memory_order_relaxed);
}

// 注册 tag(幂等):返回位 id。在 TRACELOG 宏里以函数内 static 调用,
// 每个调用点只执行一次。
inline uint32_t reg(const char * name) {
    init_once();
    std::lock_guard<std::mutex> lk(st().mtx);
    uint32_t id = (uint32_t) st().tag_names.size();
    st().tag_names.push_back(name);

    uint64_t bit = 0;
    if (st().tags_all) {
        bit = 1;
    } else {
        const std::string pat = "," + std::string(name) + ",";
        bit = st().tags_csv.find(pat) != std::string::npos ? 1 : 0;
    }
    uint64_t mask = st().mask.load(std::memory_order_relaxed);
    mask = (mask & ~(1ull << id)) | (bit << id);
    st().mask.store(mask, std::memory_order_relaxed);
    return id;
}

// 是否启用(热路径判断用;与宏配套)
inline bool on(uint32_t id) {
    return st().on.load(std::memory_order_relaxed) &&
           ((st().mask.load(std::memory_order_relaxed) >> id) & 1ull);
}

inline void vemit(uint32_t id, const char * fmt, va_list ap) {
    if (!on(id)) {
        return;
    }

    char line[kLineMax];
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    if (n <= 0) {
        return;
    }
    if (n >= (int) sizeof(line)) {
        n = (int) sizeof(line) - 1;
        line[n] = '\0';
    }

    std::lock_guard<std::mutex> lk(st().mtx);
    auto & s = st().sites[(uint64_t) (uintptr_t) fmt];
    if (s.last == line) {
        s.dups++;                          // 完全重复 → 节流
        return;
    }
    if (s.dups > 0) {
        st().buf += "[trace] [dup x" + std::to_string(s.dups) + "]\n";
        s.dups = 0;
    }
    st().buf += line;
    st().buf += '\n';
    s.last.assign(line, (size_t) n);

    if (st().buf.size() >= kFlushCap) {    // 兜底:超限时就地 flush
        fwrite(st().buf.data(), 1, st().buf.size(), stderr);
        st().buf.clear();
    }
}

inline void emit(uint32_t id, const char * fmt, ...) __attribute__((format(printf, 2, 3)));
inline void emit(uint32_t id, const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vemit(id, fmt, ap);
    va_end(ap);
}

// 安全点批量输出(缓冲为空时近零开销)
inline void flush() {
    if (st().buf.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lk(st().mtx);
    if (!st().buf.empty()) {
        fwrite(st().buf.data(), 1, st().buf.size(), stderr);
        st().buf.clear();
    }
}

} // namespace tracelog

// 函数体内使用;tag 名为字符串字面量;fmt 必须是字面量(节流 key 用其地址)
#define TRACELOG(tag, fmt, ...)                                                        \
    do {                                                                               \
        static const uint32_t tracelog_tag_id = tracelog::reg(tag);                    \
        if (tracelog::on(tracelog_tag_id)) {                                           \
            tracelog::emit(tracelog_tag_id, fmt, ##__VA_ARGS__);                       \
        }                                                                              \
    } while (0)

#else // !__cplusplus
#define TRACELOG(tag, fmt, ...) do {} while (0)
#endif

