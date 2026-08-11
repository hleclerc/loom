#include <sdot/support/string/to_string.h>
#include <sdot/support/INFO.h>
#include <stdexcept>
#include <iostream>
#include <vector>

struct TestFunc {
    using              Func    = void();

    /* */              TestFunc( std::string name, std::string tags, Func *func ) : name( name ), tags( tags ), func( func ) { prev_test_func = last_test_func; last_test_func = this; }

    static std::string current_section;
    static TestFunc*   last_test_func;
    TestFunc*          prev_test_func;
    std::string        name;
    std::string        tags;
    Func*              func;

    static bool matches( const TestFunc *test, const std::vector<std::string>& filter_names, const std::vector<std::string>& filter_tags ) {
        if ( ! filter_names.empty() ) {
            bool match = false;
            for (const auto& name : filter_names) {
                if (test->name == name || test->name.find(name) != std::string::npos) {
                    match = true;
                    break;
                }
            }
            if (!match) return false;
        }

        if ( ! filter_tags.empty() ) {
            bool match = false;
            for (const auto& tag : filter_tags) {
                if (test->tags.find(tag) != std::string::npos) {
                    match = true;
                    break;
                }
            }
            if (!match) return false;
        }

        return true;
    }
};

TestFunc *TestFunc::last_test_func = nullptr;
std::string TestFunc::current_section;

struct SectionScope {
    SectionScope( const std::string& name ) : prev( TestFunc::current_section ) {
        TestFunc::current_section = name;
    }
    ~SectionScope() {
        TestFunc::current_section = prev;
    }
    operator bool() const {
        return true;
    }

    std::string prev;
};


#define CHECK( condition ) \
    do { \
        ++check_count; \
        if ( ! ( condition ) ) { \
            std::string msg = "CHECK failed: " #condition; \
            if (!TestFunc::current_section.empty()) \
                msg += " (section: " + TestFunc::current_section + ")"; \
            failures.push_back({__FILE__, __LINE__, msg}); \
            throw std::runtime_error( msg ); \
        } \
    } while( false )

// concaténation avec expansion (sinon __LINE__ n'est pas développé -> collisions)
#define _TM_CAT2( a, b ) a##b
#define _TM_CAT( a, b ) _TM_CAT2( a, b )

#define TEST_CASE( name, tags ) \
    static void _TM_CAT( _test_func_, __LINE__ )(); \
    static TestFunc _TM_CAT( _test_obj_, __LINE__ )(name, tags, _TM_CAT( _test_func_, __LINE__ )); \
    static void _TM_CAT( _test_func_, __LINE__ )()

#define SECTION( name ) \
    if ( SectionScope _TM_CAT( _section_scope_, __LINE__ ){ name } )

#define CHECK_REPR( A, B ) \
    do { \
        ++check_count; \
        auto _repr_a = to_string( A ); \
        auto _repr_b = to_string( B ); \
        if ( _repr_a != _repr_b ) { \
            std::string _msg = "CHECK_REPR failed:\n    " #A " = " + _repr_a + "\n    " #B " = " + _repr_b; \
            if (!TestFunc::current_section.empty()) \
                _msg += " (section: " + TestFunc::current_section + ")"; \
            failures.push_back({__FILE__, __LINE__, _msg}); \
            throw std::runtime_error( _msg ); \
        } \
    } while( false )

namespace {
    constexpr const char* GREEN = "\033[92m";
    constexpr const char* RED = "\033[91m";
    constexpr const char* RESET = "\033[0m";

    int check_count = 0;

    struct CheckFailure {
        std::string file;
        int line;
        std::string message;
    };

    std::vector<CheckFailure> failures;
}

int main( int argc, char **argv ) {
    // parse args
    std::vector<std::string> filter_names, filter_tags;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find('[') != std::string::npos) {
            filter_tags.push_back(arg);
        } else {
            filter_names.push_back(arg);
        }
    }

    // run the tests
    int passed = 0, failed = 0;
    for (TestFunc* test = TestFunc::last_test_func; test != nullptr; test = test->prev_test_func) {
        if (!TestFunc::matches(test, filter_names, filter_tags))
            continue;

        try {
            test->func();
            std::cout << GREEN << "PASS: " << RESET << test->name << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << RED << "FAIL: " << RESET << test->name << " - " << e.what() << std::endl;
            failed++;
        }
    }

    if ( failed ) {
        std::cout << "\n" << GREEN << passed << " passed" << RESET << ", " << RED << failed << " failed" << RESET << " (" << check_count << " checks)\n";
        if ( !failures.empty() ) {
            std::cout << "\n" << RED << "Failed checks:" << RESET << "\n";
            for (const auto& f : failures) {
                std::cout << "  " << f.file << ":" << f.line << " - " << f.message << "\n";
            }
        }
    } else {
        std::cout << "\n" << GREEN << "All good (" << passed << " test(s), " << check_count << " checks) !" << RESET << "\n";
    }
    return failed > 0 ? 1 : 0;
}
